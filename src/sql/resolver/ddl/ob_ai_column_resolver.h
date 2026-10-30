/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "share/schema/ob_column_schema.h"
#include "share/schema/ob_table_schema.h"
#include "sql/parser/parse_node.h"

namespace oceanbase { namespace sql {
class ObSQLSessionInfo;
class ObAlterTableStmt;
class ObStmt;
struct ObResolverParams;
int resolve_ai_column_attribute(share::schema::ObColumnSchemaV2 &column, const ParseNode &node, ObSQLSessionInfo &session);
int finalize_ai_columns(share::schema::ObTableSchema &table);
int validate_ai_column_expressions(share::schema::ObTableSchema &table, ObResolverParams &params);
int finalize_alter_ai_columns(ObAlterTableStmt &statement, const share::schema::ObTableSchema &original, ObResolverParams &params);
int rewrite_ai_column_statement(ObResolverParams &params, const ParseNode &node, ObStmt *&statement, bool &handled);
bool is_ai_column_retry_resolution();
class AiColumnResolutionGuard
{
public:
  AiColumnResolutionGuard();
  ~AiColumnResolutionGuard();
private:
  bool previous_;
};
}}
