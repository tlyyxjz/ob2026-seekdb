/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_RESV
#include "sql/resolver/ddl/ob_ai_column_resolver.h"
#include "share/schema/ob_ai_column_schema.h"
#include "sql/resolver/ddl/ob_alter_table_stmt.h"
#include "sql/session/ob_sql_session_info.h"
#include "query/vector/ob_vector_index_util.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"
#include "sql/resolver/ob_resolver.h"
#include "sql/resolver/ob_schema_checker.h"
#include "sql/parser/ob_parser.h"
#include "sql/resolver/expr/ob_raw_expr_util.h"
#include <algorithm>
#include <set>
#include <limits>

namespace oceanbase { namespace sql {
using namespace common;
using namespace share::schema;
namespace ac = share::schema::ai_column;
namespace {
thread_local bool ai_internal_resolution = false;
std::string node_string(const ParseNode *node)
{
  if (!node || !node->str_value_) throw ac::Error("AI COLUMN: expected literal or identifier");
  return {node->str_value_, static_cast<size_t>(node->str_len_)};
}
void check(bool valid, const char *message) { if (!valid) throw ac::Error(message); }
int64_t integer(const ParseNode *node)
{
  check(node && node->type_ == T_INT, "AI COLUMN: expected an integer literal"); return node->value_;
}
std::vector<std::string> arguments(const std::string &text)
{
  std::vector<std::string> result;
  size_t start = std::string::npos; int depth = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (c == '\'' || c == '"' || c == '`') {
      char quote = c; bool closed = false;
      while (++i < text.size()) {
        if (text[i] == '\\' && quote != '`' && i + 1 < text.size()) ++i;
        else if (text[i] == quote) {
          if (i + 1 < text.size() && text[i + 1] == quote) ++i;
          else { closed = true; break; }
        }
      }
      check(closed, "AI COLUMN: unterminated expression literal");
    } else if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t end = text.find("*/", i + 2); check(end != std::string::npos, "AI COLUMN: unterminated comment"); i = end + 1;
    } else if (c == '(') {
      if (++depth == 1) start = i + 1;
    } else if (c == ',' && depth == 1) {
      result.push_back(text.substr(start, i - start)); start = i + 1;
    } else if (c == ')' && --depth == 0) {
      check(start != std::string::npos, "AI COLUMN: malformed function");
      result.push_back(text.substr(start, i - start)); return result;
    }
  }
  throw ac::Error("AI COLUMN: malformed function expression");
}
void collect_dependencies(const ParseNode *node, ac::Definition &definition, std::set<std::string> &seen, int depth = 0)
{
  check(depth < 64, "AI COLUMN: input expression exceeds nesting limit");
  if (!node) return;
  check(node->type_ != T_SELECT && node->type_ != T_REF_QUERY && node->type_ != T_OP_GET_USER_VAR
        && node->type_ != T_QUESTIONMARK, "AI COLUMN: subqueries, variables and parameters are not supported");
  if (node->type_ == T_COLUMN_REF) {
    check(node->num_child_ == 3 && !node->children_[0] && !node->children_[1],
          "AI COLUMN: dependencies must be unqualified columns");
    std::string name = node_string(node->children_[2]);
    if (seen.insert(ac::fold(name)).second) definition.dependencies.push_back({0, name});
  } else if (node->type_ == T_FUN_SYS) {
    check(node->num_child_ >= 1, "AI COLUMN: invalid input function");
    std::string name = ac::fold(node_string(node->children_[0]));
    static const std::set<std::string> forbidden = {
      "ai_complete", "ai_embed", "ai_rerank", "rand", "random", "uuid", "uuid_short", "sleep",
      "now", "sysdate", "current_timestamp", "current_date", "current_time", "connection_id", "last_insert_id"
    };
    check(!forbidden.count(name), "AI COLUMN: input expression must be deterministic and cannot call AI functions");
  }
  for (int i = 0; i < node->num_child_; ++i) collect_dependencies(node->children_[i], definition, seen, depth + 1);
}

int append_hidden(ObTableSchema &table, uint64_t id, const std::string &name, bool state)
{
  ObColumnSchemaV2 column;
  int ret = OB_SUCCESS;
  column.set_table_id(table.get_table_id()); column.set_column_id(id);
  column.set_data_type(state ? ObIntType : ObVarcharType);
  column.set_collation_type(state ? CS_TYPE_BINARY : CS_TYPE_UTF8MB4_BIN);
  column.set_data_length(state ? 8 : 64); column.set_nullable(true); column.set_is_hidden(true);
  ObObj value;
  if (state) value.set_int(static_cast<int64_t>(ac::State::Pending)); else value.set_null();
  if (OB_FAIL(column.set_column_name(ObString(name.size(), name.data())))) {
  } else if (OB_FAIL(column.set_orig_default_value(value))) {
  } else if (OB_FAIL(column.set_cur_default_value(value, false))) {
  } else if (OB_FAIL(table.add_column(column))) {
  }
  return ret;
}

int finalize_definition(ObColumnSchemaV2 &column, ObTableSchema &table, uint64_t &next_id)
{
  int ret = OB_SUCCESS;
  ac::Definition definition = ac::Definition::deserialize(std::string(column.get_ai_column_info().ptr(), column.get_ai_column_info().length()));
  check(column.is_nullable() && !column.is_rowkey_column() && !column.is_autoincrement() && !column.is_generated_column(),
        "AI COLUMN: result must be a nullable ordinary non-key column");
  for (auto &dependency : definition.dependencies) {
    const auto *source = table.get_column_schema(ObString(dependency.name.size(), dependency.name.data()));
    check(source && !source->is_hidden(), "AI COLUMN: unknown input column");
    check(!source->is_ai_column(), "AI COLUMN: dependencies on AI columns are not supported");
    dependency.column_id = source->get_column_id(); dependency.name.assign(source->get_column_name_str().ptr(), source->get_column_name_str().length());
  }
  std::set<uint64_t> unique_dependencies;
  definition.dependencies.erase(std::remove_if(definition.dependencies.begin(), definition.dependencies.end(),
      [&](const ac::Dependency &value) { return !unique_dependencies.insert(value.column_id).second; }), definition.dependencies.end());
  if (definition.state_column_id == 0) {
    definition.state_column_id = ++next_id; definition.request_column_id = ++next_id; definition.result_spec_column_id = ++next_id;
    std::string suffix = std::to_string(column.get_column_id());
    if (OB_FAIL(append_hidden(table, definition.state_column_id, "__ai_state_" + suffix, true))) {
    } else if (OB_FAIL(append_hidden(table, definition.request_column_id, "__ai_request_" + suffix, false))) {
    } else if (OB_FAIL(append_hidden(table, definition.result_spec_column_id, "__ai_spec_" + suffix, false))) {
    }
  }
  if (OB_SUCC(ret)) {
    definition.refresh_version(); std::string info = definition.serialize();
    ret = column.set_ai_column_info(ObString(info.size(), info.data()));
  }
  return ret;
}
}

int resolve_ai_column_attribute(ObColumnSchemaV2 &column, const ParseNode &node, ObSQLSessionInfo &session)
{
  int ret = OB_SUCCESS;
  UNUSED(session);
  try {
    check(!column.is_ai_column() && node.num_child_ == 2, "AI COLUMN: duplicate or invalid declaration");
    const ParseNode *function = node.children_[0];
    check(function && function->type_ == T_FUN_SYS && function->num_child_ == 2, "AI COLUMN: expected AI_COMPLETE or AI_EMBED");
    ac::Definition definition;
    std::string name = ac::fold(node_string(function->children_[0]));
    check(name == "ai_complete" || name == "ai_embed", "AI COLUMN: expected AI_COMPLETE or AI_EMBED");
    definition.function = name == "ai_embed" ? "AI_EMBED" : "AI_COMPLETE";
    const ParseNode *args = function->children_[1];
    check(args && args->num_child_ >= 2 && args->num_child_ <= 3, "AI COLUMN: invalid AI function arguments");
    check(args->children_[0] && args->children_[0]->type_ == T_VARCHAR, "AI COLUMN: model must be a nonempty string literal");
    definition.model = node_string(args->children_[0]); check(!definition.model.empty(), "AI COLUMN: empty model");
    definition.expression = node_string(&node);
    auto source_args = arguments(definition.expression);
    check(source_args.size() == static_cast<size_t>(args->num_child_), "AI COLUMN: invalid argument source text");
    definition.input_expression = source_args[1];
    std::set<std::string> dependencies;
    collect_dependencies(args->children_[1], definition, dependencies);
    if (name == "ai_embed") {
      check(column.is_collection(), "AI COLUMN: AI_EMBED requires a VECTOR result column");
      int64_t dimension = 0;
      if (OB_FAIL(share::ObVectorIndexUtil::get_vector_dim_from_extend_type_info(column.get_extended_type_info(), dimension))) return ret;
      if (args->num_child_ == 3 && integer(args->children_[2]) != dimension) {
        FORWARD_USER_ERROR_MSG(OB_ERR_INVALID_VECTOR_DIM, "inconsistent dimension: expected %ld got %ld", dimension, integer(args->children_[2]));
        return OB_ERR_INVALID_VECTOR_DIM;
      }
      check(dimension > 0, "AI COLUMN: invalid vector dimension"); definition.dimension = dimension;
      definition.result_type = "VECTOR(" + std::to_string(dimension) + ")";
      if (args->num_child_ == 3) definition.request_dimension = dimension;
    } else {
      check(args->num_child_ == 2, "AI COLUMN: AI_COMPLETE currently requires model and input arguments");
      if (column.is_json()) { definition.result_type = "JSON"; definition.type_hint = "Return exactly one valid JSON value."; }
      else if (ob_is_integer_type(column.get_data_type())) { definition.result_type = "SIGNED"; definition.type_hint = "Return exactly one base-10 integer."; }
      else if (ob_is_numeric_type(column.get_data_type())) { definition.result_type = "DOUBLE"; definition.type_hint = "Return exactly one finite number."; }
      else if ((column.get_data_type() == ObDateType || column.get_data_type() == ObMySQLDateType)) { definition.result_type = "DATE"; definition.type_hint = "Return exactly one valid date in YYYY-MM-DD format."; }
      else if ((column.get_data_type() == ObDateTimeType || column.get_data_type() == ObMySQLDateTimeType)) { definition.result_type = "DATETIME"; definition.type_hint = "Return exactly one valid datetime in YYYY-MM-DD HH:MM:SS.ffffff format."; }
      else {
        check(column.is_string_type(), "AI COLUMN: unsupported completion result type");
        definition.result_type = "CHAR"; definition.type_hint = "Return only the generated text.";
      }
    }
    if (node.children_[1]) {
      const ParseNode *options = node.children_[1]; std::set<std::string> seen;
      for (int i = 0; i < options->num_child_; ++i) {
        const ParseNode *option = options->children_[i];
        check(option && option->type_ == T_OP_EQ && option->num_child_ == 2
              && option->children_[0] && option->children_[0]->type_ == T_COLUMN_REF, "AI COLUMN: options must be named");
        std::string key = ac::fold(node_string(option->children_[0]->children_[2]));
        check(seen.insert(key).second, "AI COLUMN: duplicate option");
        if (key == "sync_mode") {
          check(option->children_[1]->type_ == T_VARCHAR && ac::fold(node_string(option->children_[1])) == "async",
                "AI COLUMN: only sync_mode='async' is supported");
        } else if (key == "max_retries") {
          definition.max_retries = integer(option->children_[1]);
          check(definition.max_retries >= 0 && definition.max_retries <= std::numeric_limits<int32_t>::max(), "AI COLUMN: max_retries must be a nonnegative 32-bit integer");
        } else throw ac::Error("AI COLUMN: unknown option");
      }
    }
    ObArenaAllocator model_allocator("AiColumnDDL");
    ObAIFuncExprInfo *model_info = nullptr;
    if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(model_allocator,
        ObString(definition.model.size(), definition.model.data()), model_info))) return ret;
    check(model_info && model_info->type_ == (definition.dimension
        ? share::EndpointType::DENSE_EMBEDDING : share::EndpointType::COMPLETION),
        "AI COLUMN: registered model type does not match the AI function");
    definition.normalized_expression = definition.function + "(" + ac::literal(definition.model) + "," + definition.input_expression;
    if (name == "ai_embed") definition.normalized_expression += "," + std::to_string(definition.dimension);
    else definition.normalized_expression += ",JSON_OBJECT('system','" + definition.type_hint + "')";
    definition.normalized_expression += ")";
    definition.refresh_version(); std::string info = definition.serialize();
    ret = column.set_ai_column_info(ObString(info.size(), info.data()));
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) { ret = OB_NOT_SUPPORTED; FORWARD_USER_ERROR_MSG(OB_NOT_SUPPORTED, "Not supported feature or function"); }
  return ret;
}

int finalize_ai_columns(ObTableSchema &table)
{
  int ret = OB_SUCCESS;
  try {
    uint64_t next_id = table.get_max_used_column_id();
    std::vector<uint64_t> columns;
    for (int64_t i = 0; i < table.get_column_count(); ++i) {
      const auto *column = table.get_column_schema_by_idx(i);
      if (column && column->is_ai_column()) columns.push_back(column->get_column_id());
    }
    check(columns.size() <= 64, "AI COLUMN: a table supports at most 64 AI columns");
    for (uint64_t id : columns) {
      auto *column = table.get_column_schema(id);
      if (OB_FAIL(finalize_definition(*column, table, next_id))) break;
    }
    bool visible_key = false;
    for (int64_t i = 0; i < table.get_rowkey_info().get_size(); ++i) {
      const auto *key = table.get_column_schema(table.get_rowkey_info().get_column(i)->column_id_);
      visible_key |= key && !key->is_hidden();
    }
    if (OB_SUCC(ret) && !columns.empty() && !visible_key
        && !table.get_column_schema(ObString::make_string(ac::ROW_ID_COLUMN))) {
      ret = append_hidden(table, ++next_id, ac::ROW_ID_COLUMN, false);
    }
    table.set_max_used_column_id(next_id);
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) { ret = OB_NOT_SUPPORTED; FORWARD_USER_ERROR_MSG(OB_NOT_SUPPORTED, "Not supported feature or function"); }
  return ret;
}

int validate_ai_column_expressions(ObTableSchema &table, ObResolverParams &params)
{
  int ret = OB_SUCCESS;
  try {
    for (int64_t i = 0; OB_SUCC(ret) && i < table.get_column_count(); ++i) {
      const auto *column = table.get_column_schema_by_idx(i);
      if (!column || !column->is_ai_column()) continue;
      auto definition = ac::Definition::deserialize(std::string(column->get_ai_column_info().ptr(), column->get_ai_column_info().length()));
      ObRawExpr *expression = nullptr;
      ret = ObRawExprUtils::build_generated_column_expr(nullptr,
          ObString(definition.input_expression.size(), definition.input_expression.data()),
          *params.expr_factory_, *params.session_info_, table, expression, params.schema_checker_);
    }
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &) { ret = OB_INVALID_ARGUMENT; }
  return ret;
}

int finalize_alter_ai_columns(ObAlterTableStmt &statement, const ObTableSchema &original, ObResolverParams &params)
{
  int ret = OB_SUCCESS;
  ObTableSchema combined;
  if (OB_FAIL(combined.assign(original))) return ret;
  auto &alter = statement.get_alter_table_schema();
  try {
    uint64_t next_id = original.get_max_used_column_id();
    for (auto it = alter.column_begin(); OB_SUCC(ret) && it != alter.column_end(); ++it) {
      auto *column = static_cast<AlterColumnSchema *>(*it);
      if (!column) continue;
      if (column->alter_type_ == OB_DDL_ADD_COLUMN) {
        // The DDL service assigns IDs in action order, including ordinary columns.
        column->set_column_id(++next_id);
        OZ(combined.add_column(*column));
      } else if (column->is_ai_column()) {
        check(column->alter_type_ == OB_DDL_MODIFY_COLUMN, "AI COLUMN: unsupported column definition change");
        const auto *previous = original.get_column_schema(column->get_column_name_str());
        check(previous && previous->is_ai_column(), "AI COLUMN: use ADD to introduce an AI column");
        auto definition = ac::Definition::deserialize(std::string(column->get_ai_column_info().ptr(), column->get_ai_column_info().length()));
        auto old = ac::Definition::deserialize(std::string(previous->get_ai_column_info().ptr(), previous->get_ai_column_info().length()));
        definition.state_column_id = old.state_column_id;
        definition.request_column_id = old.request_column_id;
        definition.result_spec_column_id = old.result_spec_column_id;
        column->set_column_id(previous->get_column_id());
        std::string info = definition.serialize(); OZ(column->set_ai_column_info(ObString(info.size(), info.data())));
        auto *replacement = combined.get_column_schema(previous->get_column_id());
        OZ(replacement->assign(*column));
      }
    }
    combined.set_max_used_column_id(next_id);
    if (OB_SUCC(ret)) ret = finalize_ai_columns(combined);
    if (OB_SUCC(ret)) ret = validate_ai_column_expressions(combined, params);
    if (OB_SUCC(ret)) {
      for (auto it = alter.column_begin(); it != alter.column_end(); ++it) {
        auto *column = static_cast<AlterColumnSchema *>(*it);
        if (column && column->is_ai_column()) {
          const auto *resolved = combined.get_column_schema(column->get_column_id());
          OZ(column->set_ai_column_info(resolved->get_ai_column_info()));
        }
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < combined.get_column_count(); ++i) {
        const auto *column = combined.get_column_schema_by_idx(i);
        if (column->get_column_id() > next_id) {
          AlterColumnSchema hidden; OZ(hidden.assign(*column)); hidden.alter_type_ = OB_DDL_ADD_COLUMN;
          OZ(statement.add_column(hidden));
        }
      }
      alter.set_max_used_column_id(combined.get_max_used_column_id());
    }
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) { ret = OB_NOT_SUPPORTED; FORWARD_USER_ERROR_MSG(OB_NOT_SUPPORTED, "Not supported feature or function"); }
  return ret;
}

AiColumnResolutionGuard::AiColumnResolutionGuard() : previous_(ai_internal_resolution)
{ ai_internal_resolution = true; }
AiColumnResolutionGuard::~AiColumnResolutionGuard() { ai_internal_resolution = previous_; }
bool is_ai_column_retry_resolution() { return ai_internal_resolution; }

namespace {
std::string retry_predicate(const std::string &text, const std::string &state)
{
  std::string output;
  const std::string status = "(CASE " + ac::identifier(state) + " WHEN 0 THEN 'PENDING' WHEN 1 THEN 'READY' ELSE 'FAILED' END)";
  for (size_t i = 0; i < text.size();) {
    size_t start = i;
    if (text[i] == '\'' || text[i] == '"' || text[i] == '`') {
      char quote = text[i++]; bool closed = false;
      while (i < text.size()) {
        char c = text[i++];
        if (c == '\\' && quote != '`' && i < text.size()) ++i;
        else if (c == quote) {
          if (i < text.size() && text[i] == quote) ++i;
          else { closed = true; break; }
        }
      }
      check(closed, "AI COLUMN RETRY: unterminated literal"); output.append(text, start, i - start);
    } else if ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z') || text[i] == '_') {
      while (i < text.size() && ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z')
             || (text[i] >= '0' && text[i] <= '9') || text[i] == '_' || text[i] == '$')) ++i;
      std::string token = text.substr(start, i - start);
      output += ac::fold(token) == "status" ? status : token;
    } else output.push_back(text[i++]);
  }
  return output;
}
}

int rewrite_ai_column_statement(ObResolverParams &params, const ParseNode &node, ObStmt *&statement, bool &handled)
{
  int ret = OB_SUCCESS;
  handled = false;
  const ParseNode *relation = nullptr, *retry = nullptr;
  const bool show = node.type_ == T_SHOW_AI_COLUMNS;
  if (show) {
    handled = true;
    if (node.num_child_ != 1) return OB_ERR_UNEXPECTED;
    relation = node.children_[0];
  } else if (node.type_ == T_ALTER_TABLE && node.num_child_ == 4 && node.children_[1]) {
    const ParseNode *actions = node.children_[1];
    for (int i = 0; i < actions->num_child_; ++i) {
      if (actions->children_[i] && actions->children_[i]->type_ == T_RETRY_AI_COLUMN) retry = actions->children_[i];
    }
    if (retry) {
      handled = true;
      if (actions->num_child_ != 1) {
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "combining AI COLUMN RETRY with another ALTER action");
        return OB_NOT_SUPPORTED;
      }
      relation = node.children_[0];
    }
  }
  if (!handled) return OB_SUCCESS;
  try {
    check(relation && relation->num_child_ >= 2, "AI COLUMN: invalid table reference");
    ObString database = params.session_info_->get_database_name();
    if (relation->children_[0]) database.assign_ptr(relation->children_[0]->str_value_, relation->children_[0]->str_len_);
    const std::string table_name = node_string(relation->children_[1]);
    const ObTableSchema *table = nullptr;
    if (OB_FAIL(params.schema_checker_->get_table_schema(database, ObString(table_name.size(), table_name.data()), false, table))) return ret;
    check(table, "AI COLUMN: table does not exist");
    std::string sql;
    if (show) {
      sql = "SELECT column_name,expression,normalized_expr,sync_mode,max_retries FROM oceanbase.__all_ai_column WHERE table_id="
          + std::to_string(table->get_table_id()) + " ORDER BY column_id";
    } else {
      const std::string name = node_string(retry->children_[0]);
      const auto *column = table->get_column_schema(ObString(name.size(), name.data()));
      if (!column || !column->is_ai_column()) {
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "retrying a column that is not an AI column");
        return OB_NOT_SUPPORTED;
      }
      auto definition = ac::Definition::deserialize(std::string(column->get_ai_column_info().ptr(), column->get_ai_column_info().length()));
      const auto *request = table->get_column_schema(definition.request_column_id);
      const auto *state = table->get_column_schema(definition.state_column_id);
      check(request && state, "AI COLUMN: missing state columns");
      sql = "UPDATE " + ac::identifier(std::string(database.ptr(), database.length())) + "." + ac::identifier(table_name)
          + " SET " + ac::identifier(std::string(request->get_column_name_str().ptr(), request->get_column_name_str().length())) + "=UUID()";
      if (retry->children_[1]) {
        sql += " WHERE " + retry_predicate(node_string(retry->children_[1]),
            std::string(state->get_column_name_str().ptr(), state->get_column_name_str().length()));
      }
    }
    AiColumnResolutionGuard guard;
    ParseResult parsed{};
    ObParser parser(*params.allocator_, params.session_info_->get_sql_mode());
    if (OB_FAIL(parser.parse(ObString(sql.size(), sql.data()), parsed))) return ret;
    ObResolver resolver(params);
    ret = resolver.resolve(ObResolver::IS_NOT_PREPARED_STMT, *parsed.result_tree_->children_[0], statement);
    params.query_ctx_->get_query_hint_for_update().get_global_hint().merge_plan_cache_hint(OB_USE_PLAN_CACHE_NONE);
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) { ret = OB_NOT_SUPPORTED; FORWARD_USER_ERROR_MSG(OB_NOT_SUPPORTED, "Not supported feature or function"); }
  return ret;
}
}}
