/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include <gtest/gtest.h>
#include "sql/parser/ob_parser.h"
#include "lib/allocator/ob_allocator.h"
namespace oceanbase { namespace sql { namespace {
const ParseNode *find_ai_node(const ParseNode *node, ObItemType type)
{
  if (!node || node->type_==type) return node;
  for (int i=0; i<node->num_child_; ++i) if (const auto *found=find_ai_node(node->children_[i],type)) return found;
  return nullptr;
}
TEST(AiColumnParser, declarations_and_definition_metadata)
{
  const char *queries[]={
    "CREATE TABLE docs(id BIGINT PRIMARY KEY, body TEXT, tags JSON AI COLUMN (AI_COMPLETE('m',CONCAT('Tags: ',body))) WITH(sync_mode='async',max_retries=2))",
    "ALTER TABLE docs ADD COLUMN emb VECTOR(4) AI COLUMN(AI_EMBED('e',body)) WITH(sync_mode='async')",
    "ALTER TABLE docs MODIFY COLUMN tags JSON AI COLUMN(AI_COMPLETE('m',body)) WITH(max_retries=0)"};
  for (const auto *sql:queries) {
    common::ObArenaAllocator allocator; ObParser parser(allocator,0); ParseResult result{};
    ASSERT_EQ(common::OB_SUCCESS,parser.parse(common::ObString(sql),result)) << sql;
    const auto *ai=find_ai_node(result.result_tree_,T_AI_COLUMN_DEFINITION);
    ASSERT_NE(nullptr,ai); EXPECT_EQ(2,ai->num_child_); ASSERT_NE(nullptr,ai->str_value_);
    EXPECT_TRUE(ai->is_tree_not_param_); parser.free_result(result);
  }
}
TEST(AiColumnParser, retry_predicate_and_show)
{
  const char *queries[]={"ALTER TABLE docs RETRY AI COLUMN tags WHERE status='FAILED'", "ALTER TABLE docs RETRY AI COLUMN tags WHERE `status`='active'", "ALTER TABLE docs RETRY AI COLUMN tags"};
  for (const auto *sql:queries) {
    common::ObArenaAllocator allocator; ObParser parser(allocator,0); ParseResult result{};
    ASSERT_EQ(common::OB_SUCCESS,parser.parse(common::ObString(sql),result)) << sql;
    ASSERT_NE(nullptr,find_ai_node(result.result_tree_,T_RETRY_AI_COLUMN)); parser.free_result(result);
  }
  common::ObArenaAllocator allocator; ObParser parser(allocator,0); ParseResult result{};
  ASSERT_EQ(common::OB_SUCCESS,parser.parse(common::ObString("SHOW AI COLUMNS FROM docs"),result));
  ASSERT_NE(nullptr,find_ai_node(result.result_tree_,T_SHOW_AI_COLUMNS)); parser.free_result(result);
}
}}}
