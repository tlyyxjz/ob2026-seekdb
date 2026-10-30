/* Copyright (c) 2026 OceanBase.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 */

#include <gtest/gtest.h>
#include <limits>
#include <string>
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

using namespace oceanbase::common;

namespace oceanbase {
namespace sql {
namespace {
class AIProviderEnhancementTest : public testing::Test
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

TEST_F(AIProviderEnhancementTest, embeddings_are_restored_to_input_order)
{
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, parse(
      R"({"data":[{"index":1,"embedding":[2,3]},{"index":0,"embedding":[0,1]}]})"), result));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::validate_embedding_result(allocator_, result, 2));
  ObIJsonBase *first = nullptr;
  ObIJsonBase *element = nullptr;
  ASSERT_EQ(OB_SUCCESS, result->get_array_element(0, first));
  ASSERT_EQ(OB_SUCCESS, first->get_array_element(0, element));
  EXPECT_EQ(0, element->get_int());
  EXPECT_NE(OB_SUCCESS, ObAIFuncUtils::validate_embedding_result(allocator_, result, 3));
}

TEST_F(AIProviderEnhancementTest, malformed_embeddings_fail_without_unsafe_casts)
{
  const char *responses[] = {
    R"({"data":{}})", R"({"data":[]})",
    R"({"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]})",
    R"({"data":[{"index":1,"embedding":[1]}]})",
    R"({"data":[{"index":-1,"embedding":[1]}]})",
    R"({"data":[{"index":0.5,"embedding":[1]}]})",
    R"({"data":[{"embedding":[1]}]})",
    R"({"data":[{"index":0,"embedding":1}]})",
    R"({"data":[{"index":0,"embedding":[]}]})",
    R"({"data":[{"index":0,"embedding":[true]}]})",
    R"({"data":[{"index":0,"embedding":[null]}]})",
    R"({"data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[1,2]}]})"
  };
  ObOpenAIUtils::ObOpenAIEmbed provider;
  for (const char *response : responses) {
    ObIJsonBase *result = nullptr;
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, parse(response), result));
  }
}

TEST_F(AIProviderEnhancementTest, non_finite_vector_elements_are_rejected)
{
  ObJsonArray *result = OB_NEWx(ObJsonArray, &allocator_, &allocator_);
  ObJsonArray *vector = OB_NEWx(ObJsonArray, &allocator_, &allocator_);
  ObJsonDouble *infinity = OB_NEWx(ObJsonDouble, &allocator_, std::numeric_limits<double>::infinity());
  ASSERT_NE(nullptr, result);
  ASSERT_NE(nullptr, vector);
  ASSERT_NE(nullptr, infinity);
  ASSERT_EQ(OB_SUCCESS, vector->append(infinity));
  ASSERT_EQ(OB_SUCCESS, result->append(vector));
  EXPECT_NE(OB_SUCCESS, ObAIFuncUtils::validate_embedding_result(allocator_, result));
}

TEST_F(AIProviderEnhancementTest, completion_requires_a_nonempty_string)
{
  const char *responses[] = {
    R"({"choices":[]})", R"({"choices":[{"message":{"content":null}}]})",
    R"({"choices":[{"message":{"content":[]}}]})", R"({"choices":[{"message":{"content":""}}]})"
  };
  ObOpenAIUtils::ObOpenAIComplete provider;
  for (const char *response : responses) {
    ObIJsonBase *result = nullptr;
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, parse(response), result));
  }
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_SUCCESS, provider.parse_output(allocator_, parse(R"({"choices":[{"message":{"content":"ok"}}]})"), result));
}

TEST_F(AIProviderEnhancementTest, dashscope_validates_envelopes_and_restores_text_indices)
{
  ObDashscopeUtils::ObDashscopeComplete complete;
  ObDashscopeUtils::ObDashscopeEmbed embed;
  ObIJsonBase *result = nullptr;
  const char *invalid_complete[] = {
    R"({"output":[]})", R"({"output":{"choices":{}}})",
    R"({"output":{"choices":[{"message":{"content":null}}]}})",
    R"({"output":{"choices":[{"message":{"content":""}}]}})"
  };
  for (const char *raw : invalid_complete) {
    EXPECT_NE(OB_SUCCESS, complete.parse_output(allocator_, parse(raw), result));
  }
  EXPECT_EQ(OB_SUCCESS, complete.parse_output(allocator_, parse(
      R"({"output":{"choices":[{"message":{"content":"ok"}}]}})"), result));
  ASSERT_EQ(OB_SUCCESS, embed.parse_output(allocator_, parse(
      R"({"output":{"embeddings":[{"text_index":1,"embedding":[2]},{"text_index":0,"embedding":[1]}]}})"), result));
  ObIJsonBase *first = nullptr;
  ObIJsonBase *value = nullptr;
  ASSERT_EQ(OB_SUCCESS, result->get_array_element(0, first));
  ASSERT_EQ(OB_SUCCESS, first->get_array_element(0, value));
  EXPECT_EQ(1, value->get_int());
  const char *invalid_embed[] = {
    R"({"output":1})", R"({"output":{"embeddings":{}}})",
    R"({"output":{"embeddings":[1]}})",
    R"({"output":{"embeddings":[{"embedding":[true]}]}})",
    R"({"output":{"embeddings":[{"text_index":1,"embedding":[1]}]}})"
  };
  for (const char *raw : invalid_embed) {
    EXPECT_NE(OB_SUCCESS, embed.parse_output(allocator_, parse(raw), result));
  }
}

TEST_F(AIProviderEnhancementTest, ollama_uses_native_nonstreaming_generation_and_embeddings)
{
  ObOllamaUtils::ObOllamaComplete complete;
  ObOllamaUtils::ObOllamaEmbed embed;
  ObString empty_key;
  ObArray<ObString> headers;
  ASSERT_EQ(OB_SUCCESS, complete.get_header(allocator_, empty_key, headers));
  ASSERT_FALSE(headers.empty());
  ObString model("model"), system("system"), input("hello");
  ObJsonObject *body = nullptr;
  ASSERT_EQ(OB_SUCCESS, complete.get_body(allocator_, model, system, input,
      parse(R"({"stream":true,"temperature":0.2,"max_tokens":16})"), body));
  ASSERT_NE(nullptr, body->get_value("stream"));
  EXPECT_FALSE(body->get_value("stream")->get_boolean());
  ASSERT_EQ(ObJsonNodeType::J_OBJECT, body->get_value("options")->json_type());
  EXPECT_NE(nullptr, static_cast<ObJsonObject *>(body->get_value("options"))->get_value("num_predict"));
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_SUCCESS, complete.parse_output(allocator_, parse(R"({"response":"hello"})"), result));
  EXPECT_NE(OB_SUCCESS, complete.parse_output(allocator_, parse(R"({"response":null})"), result));
  ObArray<ObString> inputs;
  ASSERT_EQ(OB_SUCCESS, inputs.push_back(input));
  ASSERT_EQ(OB_SUCCESS, inputs.push_back(system));
  ASSERT_EQ(OB_SUCCESS, embed.get_body(allocator_, model, inputs, nullptr, body));
  EXPECT_EQ(2, body->get_value("input")->element_count());
  EXPECT_EQ(OB_SUCCESS, embed.parse_output(allocator_, parse(R"({"embeddings":[[0,1],[1,0]]})"), result));
  EXPECT_NE(OB_SUCCESS, embed.parse_output(allocator_, parse(R"({"embeddings":[1]})"), result));
}

TEST_F(AIProviderEnhancementTest, rerank_covers_every_index_and_sorts_ties_by_original_position)
{
  ObJsonArray *result = nullptr;
  ObJsonObject *response = parse(R"({"results":[{"index":2,"relevance_score":0.9},{"index":0,"relevance_score":0.9},{"index":1,"relevance_score":0}]})");
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::normalize_rerank_result(allocator_, response->get_value("results"), 3, result));
  ASSERT_EQ(3, result->element_count());
  EXPECT_EQ(0, static_cast<ObJsonObject *>(result->get_value(0))->get_value("index")->get_int());
  EXPECT_EQ(2, static_cast<ObJsonObject *>(result->get_value(1))->get_value("index")->get_int());
  const char *invalid[] = {
    R"({"results":[{"index":0,"relevance_score":1},{"index":0,"relevance_score":2}]})",
    R"({"results":[{"index":0,"relevance_score":1},{"index":2,"relevance_score":2}]})",
    R"({"results":[{"index":0,"relevance_score":true},{"index":1,"relevance_score":2}]})",
    R"({"results":[{"index":0,"relevance_score":1}]})"
  };
  for (const char *raw : invalid) {
    EXPECT_NE(OB_SUCCESS, ObAIFuncUtils::normalize_rerank_result(allocator_, parse(raw)->get_value("results"), 2, result));
  }
}

TEST_F(AIProviderEnhancementTest, ollama_endpoint_allows_empty_key_and_case_preserving_provider_updates)
{
  share::ObAiModelEndpointInfo endpoint;
  ObString name("native-endpoint");
  ASSERT_EQ(OB_SUCCESS, endpoint.parse_from_json_base(allocator_, name, *parse(
      R"({"scope":"all","ai_model_name":"model","url":"http://localhost/api/generate","access_key":"","provider":"oLlAmA"})")));
  EXPECT_EQ(OB_SUCCESS, endpoint.merge_delta_endpoint(allocator_, *parse(R"({"provider":"Ollama"})")));
  EXPECT_NE(OB_SUCCESS, endpoint.merge_delta_endpoint(allocator_, *parse(R"({"provider":"openai"})")));
  ObAIFuncIComplete *complete = nullptr;
  ObAIFuncIEmbed *embed = nullptr;
  EXPECT_EQ(OB_SUCCESS, ObAIFuncUtils::get_complete_provider(allocator_, ObString("oLlAmA"), complete));
  EXPECT_EQ(OB_SUCCESS, ObAIFuncUtils::get_embed_provider(allocator_, ObString("ollama"), embed));
  EXPECT_NE(nullptr, complete);
  EXPECT_NE(nullptr, embed);
}

TEST_F(AIProviderEnhancementTest, authorization_headers_do_not_truncate_long_keys)
{
  std::string secret(2048, 'k');
  ObString key(static_cast<int32_t>(secret.size()), secret.c_str());
  ObArray<ObString> headers;
  ASSERT_EQ(OB_SUCCESS, ObOpenAIUtils::get_header(allocator_, key, headers));
  ASSERT_EQ(2, headers.count());
  EXPECT_EQ("Authorization: Bearer " + secret, std::string(headers.at(0).ptr(), headers.at(0).length()));
}
} // namespace
} // namespace sql
} // namespace oceanbase
