/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/basic/ob_ai_column_runtime.h"
#include "sql/engine/ob_exec_context.h"
#include "sql/resolver/ob_stmt.h"
#include "sql/resolver/dml/ob_dml_stmt.h"
#include "sql/resolver/dml/ob_select_stmt.h"
#include "sql/resolver/ddl/ob_ai_column_resolver.h"
#include "share/schema/ob_ai_column_schema.h"
#include "query/session/ob_inner_sql_connection_access.h"
#include "sql/session/ob_inner_sql_connection.h"
#include <set>
#include <vector>

namespace oceanbase { namespace sql {
using namespace common;
using namespace share::schema;
namespace ac = share::schema::ai_column;
namespace {
thread_local bool ai_read_check = false;
struct ReadCheckScope {
  bool previous = ai_read_check;
  ReadCheckScope() { ai_read_check = true; }
  ~ReadCheckScope() { ai_read_check = previous; }
};
struct Reference { uint64_t table_id, column_id; std::string sql; };
int collect_ai_references(const ObDMLStmt &statement, ObSchemaGetterGuard &schema_guard,
                          std::set<std::pair<uint64_t, uint64_t>> &seen, std::vector<Reference> &references, int depth)
{
  int ret = OB_SUCCESS;
  if (depth > 64) return OB_SIZE_OVERFLOW;
  // INSERT target columns are writes, including implicit AI result slots. Only
  // its source SELECT reads columns and needs readiness checks.
  for (int64_t i = 0; !statement.is_insert_stmt() && OB_SUCC(ret) && i < statement.get_column_size(); ++i) {
    const auto *item = statement.get_column_item(i);
    if (!item || !item->expr_ || !item->expr_->is_explicited_reference()) continue;
    const auto *table_item = statement.get_table_item_by_id(item->table_id_);
    if (!table_item || !table_item->is_basic_table()) continue;
    const ObTableSchema *table = nullptr;
    if (OB_FAIL(schema_guard.get_table_schema(table_item->ref_id_, table))) break;
    if (!table) continue;
    const auto *column = table->get_column_schema(item->column_id_);
    if (!column || !column->is_ai_column() || !seen.insert({table_item->ref_id_, item->column_id_}).second) continue;
    auto definition = ac::Definition::deserialize(std::string(column->get_ai_column_info().ptr(), column->get_ai_column_info().length()));
    const auto *state = table->get_column_schema(definition.state_column_id);
    const ObDatabaseSchema *database = nullptr;
    if (OB_FAIL(schema_guard.get_database_schema(table->get_database_id(), database))) break;
    if (!state || !database) { LOG_ERROR("AI strict missing schema", "table_id", table->get_table_id(), "state_id", definition.state_column_id, KP(state), KP(database)); return OB_ERR_UNEXPECTED; }
    std::string state_name(state->get_column_name_str().ptr(), state->get_column_name_str().length());
    std::string sql = "SELECT EXISTS(SELECT 1 FROM " + ac::identifier(std::string(database->get_database_name_str().ptr(), database->get_database_name_str().length()))
        + "." + ac::identifier(std::string(table->get_table_name_str().ptr(), table->get_table_name_str().length()))
        + " WHERE " + ac::identifier(state_name) + " IS NULL OR " + ac::identifier(state_name) + "<>1 LIMIT 1) AS ai_not_ready";
    references.push_back({table_item->ref_id_, item->column_id_, std::move(sql)});
  }
  ObSEArray<ObSelectStmt *, 8> children;
  if (OB_SUCC(ret) && OB_FAIL(statement.get_child_stmts(children))) {
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < children.count(); ++i) {
    if (children.at(i)) ret = collect_ai_references(*children.at(i), schema_guard, seen, references, depth + 1);
  }
  return ret;
}
}

bool is_ai_column_read_check() { return ai_read_check; }

int check_ai_column_strict_read(ObExecContext &context)
{
  int ret = OB_SUCCESS;
  if (is_ai_column_retry_resolution()) return OB_SUCCESS;
  auto *session = context.get_my_session();
  ObObj enabled;
  if (!session) return OB_NOT_INIT;
  if (!session->is_user_session()) return OB_SUCCESS;
  if (OB_FAIL(session->get_sys_variable_by_name(ObString("ai_column_strict_read"), enabled))) return ret;
  if (enabled.is_null() || !enabled.get_int()) return OB_SUCCESS;
  auto *factory = context.get_stmt_factory();
  if (!factory || !factory->get_query_ctx() || !factory->get_query_ctx()->root_stmt_
      || !context.get_sql_ctx() || !context.get_sql_ctx()->schema_guard_) return OB_SUCCESS;
  try {
    std::set<std::pair<uint64_t, uint64_t>> seen;
    std::vector<Reference> references;
    if (OB_FAIL(collect_ai_references(*factory->get_query_ctx()->root_stmt_, *context.get_sql_ctx()->schema_guard_, seen, references, 0))) { LOG_ERROR("AI strict reference collection failed", K(ret)); return ret; }
    if (references.empty()) return OB_SUCCESS;
    sqlclient::ObISQLConnectionGuard guard;
    ObIInnerSQLConnection *connection = as_inner_sql_connection(session->get_inner_conn());
    const bool owned = connection == nullptr;
    if (owned) {
      if (OB_FAIL(query::ObInnerSQLConnectionAccess::create_connection_with_external_session(session, guard))) { LOG_ERROR("AI strict connection failed", K(ret)); return ret; }
      connection = as_inner_sql_connection(guard.get_ptr());
      if (!connection) { LOG_ERROR("AI strict invalid connection", K(ret)); return OB_ERR_UNEXPECTED; }
      session->set_inner_conn(guard.get_ptr());
    }
    struct NestedScope {
      ObSQLSessionInfo *session;
      ObIInnerSQLConnection *connection;
      sqlclient::ObISQLConnectionGuard &guard;
      bool owned, nested = false;
      ObSQLSessionInfo::StmtSavedValue saved_session;
      ObIInnerSQLConnection::SavedValue saved_connection;
      int finish() {
        int code = OB_SUCCESS;
        if (nested) { nested = false; code = connection->end_nested_session(saved_session, saved_connection); }
        if (owned) { session->set_inner_conn(nullptr); guard.reset(); owned = false; }
        return code;
      }
      ~NestedScope() { finish(); }
    } scope{session, connection, guard, owned, false, {}, {}};
    if (OB_FAIL(connection->begin_nested_session(scope.saved_session, scope.saved_connection, true))) {
    } else {
      scope.nested = true;
      AiColumnResolutionGuard access;
      ReadCheckScope nested_read;
      for (const auto &reference : references) {
        ObMySQLProxy::MySQLResult result;
        if (OB_FAIL(connection->execute_read(reference.sql.c_str(), result))) break;
        if (!result.get_result()) { ret = OB_ERR_UNEXPECTED; break; }
        int next = result.get_result()->next();
        int64_t pending = 0;
        if (next == OB_SUCCESS) next = result.get_result()->get_int("ai_not_ready", pending);
        int close = result.get_result()->close();
        if (next != OB_SUCCESS) { ret = next; break; }
        if (pending != 0) {
          ret = OB_AI_COLUMN_NOT_READY;
          FORWARD_USER_ERROR_MSG(OB_AI_COLUMN_NOT_READY, "AI column has unfinished tasks for table_id=%lu, column_id=%lu", reference.table_id, reference.column_id);
          break;
        } else if (close != OB_SUCCESS) { ret = close; break; }
      }
    }
    int end = scope.finish();
    if (OB_SUCC(ret)) ret = end;
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) { ret = OB_ERR_UNEXPECTED; LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.what()); }
  if (ret != OB_SUCCESS && ret != OB_AI_COLUMN_NOT_READY) { LOG_ERROR("AI strict check failed", K(ret)); }
  return ret;
}
}}
