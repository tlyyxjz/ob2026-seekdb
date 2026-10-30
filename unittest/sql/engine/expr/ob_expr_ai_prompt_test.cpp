/* Copyright (c) 2026 OceanBase.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 */

#include <gtest/gtest.h>
#include <string>
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

using namespace oceanbase::common;

namespace oceanbase {
namespace sql {
namespace {
class AIPromptEnhancementTest : public testing::Test
{
protected:
  ObJsonObject *parse(const char *text)
  {
    ObString input(text);
    ObJsonObject *object = nullptr;
    EXPECT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_object_form_str(allocator_, input, object));
    return object;
  }
  ObArenaAllocator allocator_;
};

TEST_F(AIPromptEnhancementTest, repeated_placeholders_grow_by_actual_occurrences)
{
  std::string value(10000, 'x');
  std::string raw = "{\"template\":\"{0}/{0}/{0}\",\"args\":[\"" + value + "\"]}";
  ObJsonObject *prompt = parse(raw.c_str());
  ObString rendered;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator_, prompt, rendered));
  ASSERT_EQ(30002, rendered.length());
  EXPECT_EQ(value + "/" + value + "/" + value, std::string(rendered.ptr(), rendered.length()));
}

TEST_F(AIPromptEnhancementTest, nested_prompts_unicode_json_and_escaped_braces)
{
  ObJsonObject *prompt = parse(R"({"template":"{{{0}}} {1} {2}","args":[{"template":"你好 {0}","args":["世界"]},[1,true],{"k":"value"}]})");
  ObString rendered;
  ASSERT_TRUE(ObAIFuncPromptObjectUtils::is_valid_prompt_object(prompt));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator_, prompt, rendered));
  std::string text(rendered.ptr(), rendered.length());
  EXPECT_EQ(0U, text.find("{你好 世界}"));
  EXPECT_NE(std::string::npos, text.find("true"));
  EXPECT_NE(std::string::npos, text.find("\"k\""));
  EXPECT_NE(std::string::npos, text.find("\"value\""));
}

TEST_F(AIPromptEnhancementTest, invalid_and_overflowing_indices_are_rejected)
{
  const char *templates[] = {"{1}", "{-1}", "{x}", "{}", "{0", "}", "{999999999999999999999999999999}"};
  for (const char *text : templates) {
    ObString format(text);
    ObJsonArray *args = nullptr;
    ObJsonString *value = nullptr;
    ObJsonObject *prompt = nullptr;
    ObString argument("literal {0}");
    ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_array(allocator_, args));
    ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_string(allocator_, argument, value));
    ASSERT_EQ(OB_SUCCESS, args->append(value));
    ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::construct_prompt_object(allocator_, format, args, prompt));
    ObString rendered;
    EXPECT_NE(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator_, prompt, rendered));
  }
}

TEST_F(AIPromptEnhancementTest, argument_contents_are_not_reinterpreted_as_placeholders)
{
  ObString rendered;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, parse(R"({"template":"{0} {0}","args":["{99}\\quote\nline"]})"), rendered));
  EXPECT_EQ("{99}\\quote\nline {99}\\quote\nline", std::string(rendered.ptr(), rendered.length()));
}
} // namespace
} // namespace sql
} // namespace oceanbase
