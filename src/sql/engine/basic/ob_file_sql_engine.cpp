/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/basic/ob_file_sql_engine.h"
#include "sql/resolver/expr/ob_raw_expr.h"
#include "sql/session/ob_sql_session_info.h"
#include "share/ob_server_struct.h"
#include "share/schema/ob_schema_utils.h"

namespace oceanbase { namespace sql {
using namespace common;

bool is_file_table_expression(const ObRawExpr *expr)
{
  if (expr == nullptr || !expr->is_const_expr()) return false;
  const ObObj &value = static_cast<const ObConstRawExpr *>(expr)->get_value();
  return value.get_type() == ObVarcharType
      && file_sql::Definition::is_descriptor(value.get_string_ptr(), value.get_string_len());
}

file_sql::Definition file_table_definition(const ObRawExpr *expr)
{
  if (!is_file_table_expression(expr)) throw file_sql::Error("file_sql: invalid table expression");
  const ObObj &value = static_cast<const ObConstRawExpr *>(expr)->get_value();
  return file_sql::Definition::deserialize(std::string(value.get_string_ptr(), value.get_string_len()));
}

ObObjType file_sql_type(file_sql::Type type)
{
  switch (type) {
    case file_sql::Type::Null: return ObNullType;
    case file_sql::Type::Boolean: return ObTinyIntType;
    case file_sql::Type::Bigint: return ObIntType;
    case file_sql::Type::Double: return ObDoubleType;
    case file_sql::Type::Date: return ObDateType;
    case file_sql::Type::Datetime: return ObDateTimeType;
    default: return ObVarcharType;
  }
}

int check_file_sql_access(ObSQLSessionInfo *session, std::string &secure_root)
{
  int ret = OB_SUCCESS;
  ObObj enabled;
  ObString root;
  ObArenaAllocator allocator("FileSqlRoot");
  if (session == nullptr) ret = OB_NOT_INIT;
  else if (OB_FAIL(session->get_sys_variable_by_name(ObString("enable_file_sql"), enabled))) {
  } else if (enabled.is_null() || enabled.get_int() == 0) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "file SQL is disabled; set enable_file_sql=ON");
  } else if (GCTX.schema_service_ == nullptr) {
    ret = OB_NOT_INIT;
  } else if (OB_FAIL(share::schema::ObSchemaUtils::get_runtime_varchar_variable(
                         *GCTX.schema_service_, share::SYS_VAR_SECURE_FILE_PRIV, allocator, root))) {
  } else secure_root.assign(root.ptr() == nullptr ? "" : root.ptr(), root.length());
  return ret;
}
}} // namespace oceanbase::sql
