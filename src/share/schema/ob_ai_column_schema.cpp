/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "share/schema/ob_ai_column_schema.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace oceanbase { namespace share { namespace schema { namespace ai_column {
namespace {
void require(bool condition, const char *message) { if (!condition) throw Error(message); }
std::string string_field(const rapidjson::Value &value, const char *name)
{
  require(value.HasMember(name) && value[name].IsString(), "AI COLUMN: invalid definition string");
  return {value[name].GetString(), value[name].GetStringLength()};
}
int64_t integer_field(const rapidjson::Value &value, const char *name)
{
  require(value.HasMember(name) && value[name].IsInt64(), "AI COLUMN: invalid definition integer");
  return value[name].GetInt64();
}
}

std::string Definition::serialize() const
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  writer.StartObject();
  auto text = [&](const char *key, const std::string &value) {
    writer.Key(key); writer.String(value.data(), value.size());
  };
  auto number = [&](const char *key, int64_t value) { writer.Key(key); writer.Int64(value); };
  number("format_version", 1);
  text("expression", expression); text("input_expression", input_expression);
  text("normalized_expr", normalized_expression); text("function", function); text("model", model);
  text("type_hint", type_hint); text("result_type", result_type); text("sync_mode", "async");
  text("definition_version", definition_version);
  number("max_retries", max_retries); number("dimension", dimension); number("request_dimension", request_dimension);
  number("state_column_id", state_column_id); number("request_column_id", request_column_id);
  number("result_spec_column_id", result_spec_column_id);
  writer.Key("dependencies"); writer.StartArray();
  for (const auto &dependency : dependencies) {
    writer.StartObject(); number("column_id", dependency.column_id); text("name", dependency.name); writer.EndObject();
  }
  writer.EndArray(); writer.EndObject();
  require(buffer.GetSize() <= 65535, "AI COLUMN: definition exceeds 65535 bytes");
  return {buffer.GetString(), buffer.GetSize()};
}

Definition Definition::deserialize(const std::string &text)
{
  require(text.size() <= 65535, "AI COLUMN: oversized definition");
  rapidjson::Document json;
  json.Parse(text.data(), text.size());
  require(!json.HasParseError() && json.IsObject(), "AI COLUMN: invalid persisted definition");
  require(integer_field(json, "format_version") == 1 && string_field(json, "sync_mode") == "async",
          "AI COLUMN: unsupported definition version or mode");
  Definition value;
  value.expression = string_field(json, "expression"); value.input_expression = string_field(json, "input_expression");
  value.normalized_expression = string_field(json, "normalized_expr"); value.function = string_field(json, "function");
  value.model = string_field(json, "model"); value.type_hint = string_field(json, "type_hint");
  value.result_type = string_field(json, "result_type"); value.definition_version = string_field(json, "definition_version");
  value.max_retries = integer_field(json, "max_retries"); value.dimension = integer_field(json, "dimension");
  if (json.HasMember("request_dimension")) value.request_dimension = integer_field(json, "request_dimension");
  value.state_column_id = integer_field(json, "state_column_id");
  value.request_column_id = integer_field(json, "request_column_id");
  value.result_spec_column_id = integer_field(json, "result_spec_column_id");
  require(value.max_retries >= 0 && value.max_retries <= std::numeric_limits<int32_t>::max() && !value.model.empty(), "AI COLUMN: invalid retry limit or model");
  require(value.function == "AI_COMPLETE" || value.function == "AI_EMBED", "AI COLUMN: invalid function");
  require(value.dimension >= 0 && value.dimension <= 65535
      && (value.request_dimension == 0 || value.request_dimension == value.dimension), "AI COLUMN: invalid vector dimension");
  require(json.HasMember("dependencies") && json["dependencies"].IsArray()
          && json["dependencies"].Size() <= 1024, "AI COLUMN: invalid dependencies");
  std::unordered_set<std::string> seen;
  for (const auto &dependency : json["dependencies"].GetArray()) {
    require(dependency.IsObject(), "AI COLUMN: invalid dependency");
    Dependency item{static_cast<uint64_t>(integer_field(dependency, "column_id")), string_field(dependency, "name")};
    require(!item.name.empty() && seen.insert(fold(item.name)).second, "AI COLUMN: duplicate dependency");
    value.dependencies.push_back(std::move(item));
  }
  return value;
}

void Definition::refresh_version()
{
  definition_version.clear();
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : serialize()) { hash ^= c; hash *= 1099511628211ULL; }
  std::ostringstream out; out << std::hex << std::setfill('0') << std::setw(16) << hash;
  definition_version = out.str();
}

void Definition::remap_column_ids(const std::unordered_map<uint64_t, uint64_t> &ids)
{
  auto remap = [&](uint64_t &id) {
    if (id) { auto it = ids.find(id); require(it != ids.end(), "AI COLUMN: incomplete column-ID remap"); id = it->second; }
  };
  Definition updated = *this;
  remap(updated.state_column_id); remap(updated.request_column_id); remap(updated.result_spec_column_id);
  for (auto &dependency : updated.dependencies) remap(dependency.column_id);
  updated.refresh_version(); *this = std::move(updated);
}

std::string fold(std::string text)
{ for (char &c : text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return text; }

std::string identifier(const std::string &name)
{
  require(!name.empty() && name.find('\0') == std::string::npos, "AI COLUMN: invalid identifier");
  std::string result("`");
  for (char c : name) { result.push_back(c); if (c == '`') result.push_back(c); }
  result.push_back('`'); return result;
}

std::string literal(const std::string &text)
{
  if (text.empty()) return "''";
  std::string result("CONVERT(0x");
  const char *hex = "0123456789abcdef";
  for (unsigned char c : text) { result.push_back(hex[c >> 4]); result.push_back(hex[c & 15]); }
  result += " USING utf8mb4)"; return result;
}

std::string qualify_input(const std::string &text, const std::vector<Dependency> &dependencies,
                         const std::string &qualifier)
{
  std::unordered_set<std::string> names;
  for (const auto &dependency : dependencies) names.insert(fold(dependency.name));
  std::string result;
  auto identifier_char = [](unsigned char c) {
    return c >= 128 || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '$';
  };
  for (size_t pos = 0; pos < text.size();) {
    const size_t start = pos;
    if (text.compare(pos, 2, "/*") == 0) {
      auto end = text.find("*/", pos + 2); require(end != std::string::npos, "AI COLUMN: unterminated comment");
      pos = end + 2; result.append(text, start, pos - start);
    } else if (text[pos] == '#' || (text.compare(pos, 2, "--") == 0
        && pos + 2 < text.size() && static_cast<unsigned char>(text[pos + 2]) <= 32)) {
      auto end = text.find('\n', pos); pos = end == std::string::npos ? text.size() : end;
      result.append(text, start, pos - start);
    } else if (text[pos] == '\'' || text[pos] == '"') {
      char quote = text[pos++]; bool closed = false;
      while (pos < text.size()) {
        char c = text[pos++];
        if (c == '\\' && pos < text.size()) ++pos;
        else if (c == quote) {
          if (pos < text.size() && text[pos] == quote) ++pos;
          else { closed = true; break; }
        }
      }
      require(closed, "AI COLUMN: unterminated string");
      result.append(text, start, pos - start);
    } else if (text[pos] == '`' || identifier_char(text[pos])) {
      std::string name;
      if (text[pos] == '`') {
        ++pos; bool closed = false;
        while (pos < text.size()) {
          char c = text[pos++];
          if (c == '`') {
            if (pos < text.size() && text[pos] == '`') ++pos;
            else { closed = true; break; }
          }
          name.push_back(c);
        }
        require(closed, "AI COLUMN: unterminated identifier");
      } else {
        while (pos < text.size() && identifier_char(text[pos])) name.push_back(text[pos++]);
      }
      size_t next = pos; while (next < text.size() && static_cast<unsigned char>(text[next]) <= 32) ++next;
      if (names.count(fold(name)) && (next == text.size() || text[next] != '(')) {
        result += qualifier + "." + identifier(name);
      } else result.append(text, start, pos - start);
    } else result.push_back(text[pos++]);
  }
  return result;
}

std::string state_name(State state)
{
  switch (state) {
    case State::Pending: return "PENDING";
    case State::Ready: return "READY";
    case State::Failed: return "FAILED";
    default: throw Error("AI COLUMN: invalid state");
  }
}
}}}}
