/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include <gtest/gtest.h>
#include "share/schema/ob_ai_column_schema.h"
using namespace oceanbase::share::schema::ai_column;
namespace {
Definition example()
{
  Definition d;
  d.function="AI_COMPLETE"; d.model="registered_model";
  d.expression="AI_COMPLETE('registered_model', content)";
  d.input_expression="CONCAT('content', `content`)";
  d.result_type="JSON"; d.type_hint="Return exactly one valid JSON value.";
  d.state_column_id=20; d.request_column_id=21; d.result_spec_column_id=22;
  d.dependencies.push_back({17,"content"}); d.refresh_version(); return d;
}
TEST(AiColumnSchema, definition_round_trip)
{
  auto original=example(); auto copy=Definition::deserialize(original.serialize());
  EXPECT_EQ(original.serialize(),copy.serialize());
  EXPECT_EQ(17U,copy.dependencies.at(0).column_id);
  EXPECT_EQ(16U,copy.definition_version.size());
}
TEST(AiColumnSchema, definition_version_tracks_configuration)
{
  auto a=example(), b=a; b.max_retries++; b.refresh_version();
  EXPECT_NE(a.definition_version,b.definition_version);
  b=a; b.input_expression="title"; b.refresh_version(); EXPECT_NE(a.definition_version,b.definition_version);
}
TEST(AiColumnSchema, complete_column_remap)
{
  auto d=example(); auto previous=d.definition_version;
  d.remap_column_ids({{17,117},{20,120},{21,121},{22,122}});
  EXPECT_EQ(117U,d.dependencies[0].column_id); EXPECT_EQ(120U,d.state_column_id);
  EXPECT_EQ(121U,d.request_column_id); EXPECT_EQ(122U,d.result_spec_column_id);
  EXPECT_NE(previous,d.definition_version);
}
TEST(AiColumnSchema, incomplete_remap_is_atomic)
{
  auto d=example(); auto original=d.serialize();
  EXPECT_THROW(d.remap_column_ids({{20,120}}),Error); EXPECT_EQ(original,d.serialize());
}
TEST(AiColumnSchema, rejects_corrupt_persisted_metadata)
{
  EXPECT_THROW(Definition::deserialize("[]"),Error);
  EXPECT_THROW(Definition::deserialize("{}"),Error);
  auto d=example(); d.max_retries=-1; EXPECT_THROW(Definition::deserialize(d.serialize()),Error);
  d=example(); d.function="OTHER"; EXPECT_THROW(Definition::deserialize(d.serialize()),Error);
  d=example(); d.dependencies.push_back({18,"CONTENT"}); EXPECT_THROW(Definition::deserialize(d.serialize()),Error);
}
TEST(AiColumnSchema, preliminary_dependencies_can_be_resolved_later)
{
  auto d=example(); d.dependencies[0].column_id=0;
  EXPECT_EQ(0U,Definition::deserialize(d.serialize()).dependencies[0].column_id);
}
TEST(AiColumnSchema, SQL_escaping_preserves_literals)
{
  EXPECT_EQ("`a``b`",identifier("a`b")); EXPECT_EQ("CONVERT(0x6127005c USING utf8mb4)",literal(std::string("a\'\0\\",4)));
  EXPECT_EQ("CONCAT('content', NEW.`content`)",qualify_input("CONCAT('content', `content`)",{{17,"content"}},"NEW"));
  EXPECT_EQ("concat(NEW.`Content`, 'it\'\'s content')",qualify_input("concat(Content, 'it\'\'s content')",{{17,"content"}},"NEW"));
}
TEST(AiColumnSchema, requested_embedding_dimension_is_persisted_and_checked)
{
  auto d=example(); d.function="AI_EMBED"; d.dimension=4; d.request_dimension=4;
  auto copy=Definition::deserialize(d.serialize()); EXPECT_EQ(4,copy.request_dimension);
  d.request_dimension=3; EXPECT_THROW(Definition::deserialize(d.serialize()),Error);
}
TEST(AiColumnSchema, explicit_terminal_states)
{
  EXPECT_EQ("PENDING",state_name(State::Pending)); EXPECT_EQ("READY",state_name(State::Ready)); EXPECT_EQ("FAILED",state_name(State::Failed));
}
}
