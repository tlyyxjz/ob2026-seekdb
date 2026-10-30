/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_RESV
#include "sql/resolver/dml/ob_dml_resolver.h"
#include "sql/resolver/dml/ob_select_resolver.h"
#include "sql/engine/basic/ob_file_sql_engine.h"
#include <algorithm>
#include <cstring>
#include <set>

namespace oceanbase { namespace sql {
using namespace common;
namespace {
std::string node_text(const ParseNode *node)
{
  if (node == nullptr || node->str_value_ == nullptr) throw file_sql::Error("file_sql: expected a literal or option name");
  return std::string(node->str_value_, node->str_len_);
}
std::string folded(std::string text)
{ for (char &c : text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return text; }
std::string string_literal(const ParseNode *node)
{
  if (node == nullptr || (node->type_ != T_VARCHAR && node->type_ != T_CHAR)) throw file_sql::Error("file_sql: expected a string literal");
  return node_text(node);
}
int64_t integer_literal(const ParseNode *node)
{
  if (node != nullptr && (node->type_ == T_INT || node->type_ == T_BOOL)) return node->value_;
  if (node != nullptr && node->type_ == T_OP_NEG && node->num_child_ == 1) return -integer_literal(node->children_[0]);
  throw file_sql::Error("file_sql: expected an integer or boolean literal");
}
file_sql::Options options_from_node(const ParseNode &tree, std::string &path, std::string &function)
{
  if (tree.num_child_ != 3 || tree.children_[0] == nullptr || tree.children_[1] == nullptr)
    throw file_sql::Error("file_sql: invalid table function");
  const ParseNode &name = *tree.children_[0], &args = *tree.children_[1];
  if (name.children_[0] != nullptr || name.children_[1] == nullptr || args.num_child_ == 0)
    throw file_sql::Error("file_sql: expected an unqualified file function and path");
  function = folded(node_text(name.children_[1]));
  file_sql::Options result;
  if (function == "read_csv") result.format = file_sql::Format::Csv;
  else if (function == "read_json" || function == "read_ndjson") result.format = file_sql::Format::Json;
  else if (function == "read_parquet") result.format = file_sql::Format::Parquet;
  else if (function == "glob") result.format = file_sql::Format::Glob;
  else throw file_sql::Error("file_sql: unknown file table function");
  path = string_literal(args.children_[0]);
  std::set<std::string> seen;
  for (int i = 1; i < args.num_child_; ++i) {
    const ParseNode *option = args.children_[i];
    if (option == nullptr || option->type_ != T_OP_EQ || option->num_child_ != 2
        || option->children_[0] == nullptr || option->children_[0]->type_ != T_COLUMN_REF)
      throw file_sql::Error("file_sql: options must be named (name = value)");
    const ParseNode *key_node = option->children_[0]->children_[2], *value = option->children_[1];
    std::string key = folded(node_text(key_node));
    if (key == "delimiter") key = "delim";
    if (!seen.insert(key).second) throw file_sql::Error("file_sql: duplicate option: " + key);
    if (key == "sample_size" && result.format != file_sql::Format::Glob) result.sample_size = integer_literal(value);
    else if (key == "header" && result.format == file_sql::Format::Csv) {
      int64_t flag = integer_literal(value);
      if (flag != 0 && flag != 1) throw file_sql::Error("file_sql: header must be a boolean");
      result.header = flag != 0;
    } else if (key == "nullstr" && result.format == file_sql::Format::Csv) result.null_string = string_literal(value);
    else if ((key == "delim" || key == "quote" || key == "escape") && result.format == file_sql::Format::Csv) {
      std::string character = string_literal(value);
      if (character.size() > 1 || (key == "delim" && character.empty())) throw file_sql::Error("file_sql: delimiter, quote and escape must be one byte");
      char c = character.empty() ? 0 : character[0];
      if (key == "delim") result.delimiter = c;
      else if (key == "quote") { result.quote = c; if (seen.count("escape") == 0) result.escape = c; }
      else result.escape = c;
    } else if (key == "format" && result.format == file_sql::Format::Csv) {
      if (folded(string_literal(value)) != "csv") throw file_sql::Error("file_sql: expected CSV format");
    } else if (key == "format" && result.format == file_sql::Format::Json) {
      std::string format = folded(string_literal(value));
      if (format != "newline_delimited" && format != "ndjson" && format != "jsonl")
        throw file_sql::Error("file_sql: only newline-delimited JSON is supported");
    } else throw file_sql::Error("file_sql: unknown option: " + key);
  }
  return result;
}
} // namespace

int ObSelectResolver::resolve_file_copy_into(const ParseNode &node, ObSelectIntoItem &item)
{
  int ret = OB_SUCCESS;
  std::string root;
  if (OB_FAIL(check_file_sql_access(session_info_, root))) return ret;
  get_stmt()->get_query_ctx()->get_query_hint_for_update().get_global_hint().merge_plan_cache_hint(OB_USE_PLAN_CACHE_NONE);
  try {
    std::vector<ParseNode *> arguments{node.children_[0]};
    bool header = false;
    if (node.children_[1] != nullptr) {
      const ParseNode &options = *node.children_[1];
      for (int i = 0; i < options.num_child_; ++i) {
        if (folded(node_text(options.children_[i]->children_[0]->children_[2])) == "sample_size")
          throw file_sql::Error("file_sql: sample_size is a read option");
        arguments.push_back(options.children_[i]);
        header |= folded(node_text(options.children_[i]->children_[0]->children_[2])) == "header";
      }
    }
    ParseNode function{}, relation{}, args{}, tree{};
    function.type_ = T_IDENT; function.str_value_ = "read_csv"; function.str_len_ = 8;
    ParseNode *name_nodes[] = {nullptr, &function, nullptr};
    relation.type_ = T_RELATION_FACTOR; relation.num_child_ = 3; relation.children_ = name_nodes;
    args.type_ = T_EXPR_LIST; args.num_child_ = arguments.size(); args.children_ = arguments.data();
    ParseNode *children[] = {&relation, &args, nullptr};
    tree.num_child_ = 3; tree.children_ = children;
    std::string path, name;
    file_sql::Definition definition;
    definition.options = options_from_node(tree, path, name);
    if (!header) definition.options.header = false;
    definition.path = path;
    definition.columns.push_back({"column0", file_sql::Type::Varchar, true});
    std::string descriptor = definition.serialize();
    OZ(ob_write_string(*allocator_, ObString(descriptor.size(), descriptor.data()), item.external_properties_));
    ObString stored_path;
    OZ(ob_write_string(*allocator_, ObString(path.size(), path.data()), stored_path));
    item.outfile_name_.set_varchar(stored_path);
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) { ret = OB_INVALID_ARGUMENT; LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.what()); }
  return ret;
}

int ObDMLResolver::resolve_file_table_item(const ParseNode &tree, TableItem *&item)
{
  int ret = OB_SUCCESS;
  std::string root, path, function;
  file_sql::Options options;
  if (OB_FAIL(check_file_sql_access(session_info_, root))) return ret;
  // Local files are not schema-service dependencies. Infer each statement
  // afresh instead of caching a descriptor that can outlive its input.
  get_stmt()->get_query_ctx()->get_query_hint_for_update().get_global_hint().merge_plan_cache_hint(OB_USE_PLAN_CACHE_NONE);
  try { options = options_from_node(tree, path, function); }
  catch (const file_sql::Error &error) {
    ret = OB_ERR_PARSE_SQL;
    LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                   static_cast<int>(std::strlen(error.what())), error.what(), 1);
    return ret;
  }
  if (options.sample_size != -1) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "file_sql only supports sample_size=-1");
    return ret;
  }
  try {
    file_sql::Definition definition = file_sql::infer(path, root, options);
    std::string encoded = definition.serialize();
    ObString stored, alias;
    ObConstRawExpr *value = nullptr;
    ObObj object;
    if (OB_FAIL(ob_write_string(*allocator_, ObString(encoded.size(), encoded.data()), stored))) {
    } else if (OB_FAIL(params_.expr_factory_->create_raw_expr(T_VARCHAR, value))) {
    } else {
      object.set_varchar(stored); object.set_collation_type(CS_TYPE_BINARY);
      value->set_value(object);
      OZ(value->formalize(session_info_));
      if (OB_SUCC(ret) && (item = get_stmt()->create_table_item(*allocator_)) == nullptr) ret = OB_ALLOCATE_MEMORY_FAILED;
      if (OB_SUCC(ret)) {
        const ParseNode *name = tree.children_[2];
        if (name != nullptr) alias.assign_ptr(name->str_value_, name->str_len_);
        else OZ(get_stmt()->generate_func_table_name(*allocator_, alias));
        item->table_name_ = alias; item->alias_name_ = alias;
        item->table_id_ = generate_table_id(); item->type_ = TableItem::FUNCTION_TABLE;
        item->function_table_expr_ = value;
        OZ(get_stmt()->add_table_item(session_info_, item));
        ObSEArray<ColumnItem, 16> columns;
        OZ(resolve_file_table_columns(*item, columns));
      }
    }
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.what());
  }
  return ret;
}

int ObDMLResolver::resolve_file_table_columns(const TableItem &table, ObIArray<ColumnItem> &columns)
{
  int ret = OB_SUCCESS;
  try {
    auto definition = file_table_definition(table.function_table_expr_);
    for (size_t i = 0; OB_SUCC(ret) && i < definition.columns.size(); ++i) {
      const auto &column = definition.columns[i];
      ObString name;
      OZ(ob_write_string(*allocator_, ObString(column.name.size(), column.name.data()), name));
      ColumnItem *item = get_stmt()->get_column_item(table.table_id_, name);
      if (item == nullptr && OB_SUCC(ret)) {
        ObObjMeta meta; meta.set_type(file_sql_type(column.type));
        if (column.type == file_sql::Type::Varchar) {
          meta.set_collation_type(CS_TYPE_UTF8MB4_BIN);
          meta.set_collation_level(CS_LEVEL_IMPLICIT);
        }
        ObAccuracy accuracy = ObAccuracy::DDL_DEFAULT_ACCURACY[meta.get_type()];
        if (column.type == file_sql::Type::Varchar) accuracy.set_length(4 * 1024 * 1024);
        if (column.type == file_sql::Type::Datetime) accuracy.set_scale(column.scale);
        OZ(resolve_function_table_column_item(table, meta, accuracy, name, OB_APP_MIN_COLUMN_ID + i, item));
      }
      if (OB_SUCC(ret)) OZ(columns.push_back(*item));
    }
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) { ret = OB_INVALID_ARGUMENT; LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.what()); }
  return ret;
}
}} // namespace oceanbase::sql
