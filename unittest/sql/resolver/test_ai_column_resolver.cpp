/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include <gtest/gtest.h>
#include "sql/resolver/ddl/ob_ai_column_resolver.h"
#include "sql/parser/ob_parser.h"
#include "sql/session/ob_sql_session_info.h"
#include "share/schema/ob_ai_column_schema.h"
namespace oceanbase { namespace sql { namespace {
using namespace common;
using namespace share::schema;
const ParseNode *definition_node(const ParseNode *node)
{
  if (!node || node->type_==T_AI_COLUMN_DEFINITION) return node;
  for (int i=0;i<node->num_child_;++i) if (const auto *found=definition_node(node->children_[i])) return found;
  return nullptr;
}
TEST(AiColumnResolver, invalid_definitions_fail_before_model_access)
{
  const char *queries[]={
    "CREATE TABLE t(body TEXT, result JSON AI COLUMN(ABS(body)))",
    "CREATE TABLE t(body TEXT, result JSON AI COLUMN(AI_COMPLETE('m',body)) WITH(sync_mode='sync'))",
    "CREATE TABLE t(body TEXT, result JSON AI COLUMN(AI_COMPLETE('m',body)) WITH(max_retries=-1))",
    "CREATE TABLE t(body TEXT, result JSON AI COLUMN(AI_COMPLETE('m',body)) WITH(max_retries=1,max_retries=2))",
    "CREATE TABLE t(body TEXT, result JSON AI COLUMN(AI_COMPLETE('m',RAND())))",
    "CREATE TABLE t(body TEXT, result JSON AI COLUMN(AI_COMPLETE('m',(SELECT body FROM other_table))))",
    "CREATE TABLE t(body TEXT, result JSON AI COLUMN(AI_EMBED('m',body)))"};
  for (const auto *text:queries) {
    ObArenaAllocator allocator; ObParser parser(allocator,0); ParseResult parsed{};
    ASSERT_EQ(OB_SUCCESS,parser.parse(ObString(text),parsed)) << text;
    const auto *node=definition_node(parsed.result_tree_); ASSERT_NE(nullptr,node);
    ObColumnSchemaV2 column; column.set_data_type(ObJsonType); column.set_nullable(true);
    ObSQLSessionInfo session;
    EXPECT_EQ(OB_NOT_SUPPORTED,resolve_ai_column_attribute(column,*node,session)) << text;
    parser.free_result(parsed);
  }
}
TEST(AiColumnResolver, vector_dimension_mismatch)
{
  ObArenaAllocator allocator; ObParser parser(allocator,0); ParseResult parsed{};
  ASSERT_EQ(OB_SUCCESS,parser.parse(ObString("CREATE TABLE t(body TEXT, v VECTOR(4) AI COLUMN(AI_EMBED('m',body,3)))"),parsed));
  ObColumnSchemaV2 column; column.set_data_type(ObCollectionSQLType); column.set_nullable(true);
  ASSERT_EQ(OB_SUCCESS,column.add_type_info(ObString("VECTOR(4)")));
  ObSQLSessionInfo session;
  EXPECT_EQ(OB_ERR_INVALID_VECTOR_DIM,resolve_ai_column_attribute(column,*definition_node(parsed.result_tree_),session));
  parser.free_result(parsed);
}
void add_column(ObTableSchema &table,uint64_t id,const char *name,bool ai,const char *dependency="body")
{
  ObColumnSchemaV2 column; column.set_table_id(table.get_table_id()); column.set_column_id(id);
  column.set_data_type(ai?ObJsonType:ObVarcharType); column.set_nullable(true);
  column.set_collation_type(CS_TYPE_UTF8MB4_BIN); column.set_data_length(200);
  ASSERT_EQ(OB_SUCCESS,column.set_column_name(ObString(name)));
  if(ai) {
    ai_column::Definition d; d.function="AI_COMPLETE"; d.model="registered_model";
    d.input_expression=dependency; d.dependencies.push_back({0,dependency});
    auto text=d.serialize(); ASSERT_EQ(OB_SUCCESS,column.set_ai_column_info(ObString(text.size(),text.data())));
  }
  ASSERT_EQ(OB_SUCCESS,table.add_column(column)); table.set_max_used_column_id(id);
}
TEST(AiColumnResolver, finalizes_dependencies_and_hidden_heap_identity)
{
  ObTableSchema table; table.set_table_id(20000); table.set_table_type(USER_TABLE);
  add_column(table,16,"body",false); add_column(table,17,"tags",true);
  ASSERT_EQ(OB_SUCCESS,finalize_ai_columns(table)); EXPECT_EQ(6,table.get_column_count());
  const auto *identity=table.get_column_schema(ObString::make_string(ai_column::ROW_ID_COLUMN));
  ASSERT_NE(nullptr,identity); EXPECT_TRUE(identity->is_hidden()); EXPECT_TRUE(identity->is_ai_column_auxiliary());
  ASSERT_EQ(OB_SUCCESS,finalize_ai_columns(table)); EXPECT_EQ(6,table.get_column_count());
  const auto *result=table.get_column_schema(17); ASSERT_NE(nullptr,result); EXPECT_FALSE(result->is_hidden());
  auto d=ai_column::Definition::deserialize(std::string(result->get_ai_column_info().ptr(),result->get_ai_column_info().length()));
  EXPECT_EQ(16U,d.dependencies[0].column_id);
  for (uint64_t id:{d.state_column_id,d.request_column_id,d.result_spec_column_id}) {
    const auto *hidden=table.get_column_schema(id); ASSERT_NE(nullptr,hidden); EXPECT_TRUE(hidden->is_hidden());
    EXPECT_TRUE(hidden->is_ai_column_auxiliary());
  }
}
TEST(AiColumnResolver, rejects_AI_dependency_and_unknown_column)
{
  ObTableSchema table; table.set_table_id(20000); table.set_table_type(USER_TABLE);
  add_column(table,16,"body",false); add_column(table,17,"tags",true);
  add_column(table,18,"more_tags",true,"tags"); EXPECT_EQ(OB_NOT_SUPPORTED,finalize_ai_columns(table));
  ObTableSchema missing; missing.set_table_id(20001); missing.set_table_type(USER_TABLE);
  add_column(missing,16,"tags",true,"missing"); EXPECT_EQ(OB_NOT_SUPPORTED,finalize_ai_columns(missing));
}
}}}
