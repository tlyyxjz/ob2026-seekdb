/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include <gtest/gtest.h>
#include "share/schema/ob_column_schema.h"
#include "share/schema/ob_ai_column_schema.h"
#include <vector>
using namespace oceanbase::common;
using namespace oceanbase::share::schema;
namespace {
TEST(AiColumnNativeSchema, copy_and_serialization_own_metadata)
{
  ai_column::Definition d; d.function="AI_COMPLETE"; d.model="m"; d.input_expression="body";
  d.dependencies.push_back({17,"body"}); d.state_column_id=20; d.request_column_id=21; d.result_spec_column_id=22; d.refresh_version();
  auto info=d.serialize();
  ObColumnSchemaV2 source; source.set_table_id(20000); source.set_column_id(18);
  ASSERT_EQ(OB_SUCCESS,source.set_column_name(ObString("result")));
  source.set_data_type(ObVarcharType); source.set_data_length(100); source.set_collation_type(CS_TYPE_UTF8MB4_BIN);
  ASSERT_EQ(OB_SUCCESS,source.set_ai_column_info(ObString(info.size(),info.data())));
  ObColumnSchemaV2 copy; ASSERT_EQ(OB_SUCCESS,copy.assign(source));
  EXPECT_NE(source.get_ai_column_info().ptr(),copy.get_ai_column_info().ptr());
  std::vector<char> buffer(copy.get_serialize_size()); int64_t pos=0;
  ASSERT_EQ(OB_SUCCESS,copy.serialize(buffer.data(),buffer.size(),pos));
  ObColumnSchemaV2 restored; int64_t read=0;
  ASSERT_EQ(OB_SUCCESS,restored.deserialize(buffer.data(),pos,read)); EXPECT_EQ(pos,read);
  EXPECT_TRUE(restored.is_ai_column()); EXPECT_EQ(copy.get_ai_column_info(),restored.get_ai_column_info());
  source.reset(); EXPECT_FALSE(source.is_ai_column()); EXPECT_TRUE(copy.is_ai_column());
}
TEST(AiColumnNativeSchema, native_column_id_conversion_remaps_definition)
{
  ai_column::Definition d; d.function="AI_COMPLETE"; d.model="m";
  d.dependencies.push_back({17,"body"}); d.state_column_id=20; d.request_column_id=21; d.result_spec_column_id=22; d.refresh_version();
  auto info=d.serialize(); ObColumnSchemaV2 column; column.set_column_id(18);
  ASSERT_EQ(OB_SUCCESS,column.set_ai_column_info(ObString(info.size(),info.data())));
  hash::ObHashMap<uint64_t,uint64_t> ids; ASSERT_EQ(OB_SUCCESS,ids.create(8,"AiColTest"));
  for (uint64_t id : {17,18,20,21,22}) ASSERT_EQ(OB_SUCCESS,ids.set_refactored(id,id+100));
  ASSERT_EQ(OB_SUCCESS,column.convert_column_id(ids)); EXPECT_EQ(118U,column.get_column_id());
  auto actual=ai_column::Definition::deserialize(std::string(column.get_ai_column_info().ptr(),column.get_ai_column_info().length()));
  EXPECT_EQ(117U,actual.dependencies[0].column_id); EXPECT_EQ(120U,actual.state_column_id);
  EXPECT_EQ(121U,actual.request_column_id); EXPECT_EQ(122U,actual.result_spec_column_id);
  EXPECT_NE(d.definition_version,actual.definition_version);
}
}
