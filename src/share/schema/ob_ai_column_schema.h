/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace oceanbase { namespace share { namespace schema { namespace ai_column {
class Error : public std::runtime_error
{
public:
  explicit Error(const std::string &message) : std::runtime_error(message) {}
};
constexpr const char ROW_ID_COLUMN[] = "__ai_request_row_id";
enum class State : int64_t { Pending = 0, Ready = 1, Failed = 2 };
struct Dependency { uint64_t column_id = 0; std::string name; };
struct Definition
{
  std::string expression, input_expression, normalized_expression;
  std::string function, model, type_hint, result_type, definition_version;
  int64_t max_retries = 2, dimension = 0, request_dimension = 0;
  uint64_t state_column_id = 0, request_column_id = 0, result_spec_column_id = 0;
  std::vector<Dependency> dependencies;
  std::string serialize() const;
  static Definition deserialize(const std::string &text);
  void refresh_version();
  void remap_column_ids(const std::unordered_map<uint64_t, uint64_t> &ids);
};

std::string identifier(const std::string &name);
std::string literal(const std::string &text);
std::string fold(std::string text);
// Qualify only identifiers that were resolved as actual dependencies. Literals
// remain byte-for-byte unchanged, including quoted identifiers and escapes.
std::string qualify_input(const std::string &expression, const std::vector<Dependency> &dependencies,
                         const std::string &qualifier);
std::string state_name(State state);
}}}} // namespace oceanbase::share::schema::ai_column
