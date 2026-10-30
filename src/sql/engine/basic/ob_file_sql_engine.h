/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "sql/engine/basic/ob_file_scan_utils.h"
#include "common/object/ob_obj_type.h"

namespace oceanbase { namespace sql {
class ObRawExpr;
class ObSQLSessionInfo;
bool is_file_table_expression(const ObRawExpr *expr);
file_sql::Definition file_table_definition(const ObRawExpr *expr);
common::ObObjType file_sql_type(file_sql::Type type);
int check_file_sql_access(ObSQLSessionInfo *session, std::string &secure_root);
}} // namespace oceanbase::sql
