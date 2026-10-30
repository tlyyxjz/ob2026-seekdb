/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace oceanbase { namespace sql { namespace file_sql {

enum class Type : uint8_t { Null, Boolean, Bigint, Double, Varchar, Date, Datetime };
enum class Format : uint8_t { Csv, Json, Parquet, Glob };

class Error : public std::runtime_error
{
public:
  explicit Error(const std::string &message) : std::runtime_error(message) {}
};

struct Options
{
  Format format = Format::Csv;
  bool header = true;
  char delimiter = ',';
  char quote = '"';
  char escape = '"';
  std::string null_string;
  int64_t sample_size = -1;
};

struct Column
{
  std::string name;
  Type type = Type::Null;
  bool nullable = false;
  uint8_t scale = 0;
};

struct Cell
{
  bool is_null = true;
  std::string text;
  Type type = Type::Null;
};

struct Stamp
{
  uint64_t device = 0, inode = 0, size = 0;
  int64_t mtime = 0, mtime_ns = 0, ctime = 0, ctime_ns = 0;
  bool operator==(const Stamp &other) const;
};

struct Definition
{
  std::string path;
  std::string source_path;
  Options options;
  Stamp stamp;
  uint64_t content_hash = 0;
  std::vector<Column> columns;
  uint64_t rows = 0;
  std::string serialize() const;
  static Definition deserialize(const std::string &bytes);
  static bool is_descriptor(const char *data, int64_t length);
};

const char *type_name(Type type);
Type infer_type(const std::string &text);
Type widen(Type left, Type right);
bool valid_utf8(const std::string &text);
// Match seekdb's secure_file_priv contract: empty and literal NULL deny access.
Definition infer(const std::string &path, const std::string &secure_root, const Options &options);
std::vector<std::string> discover(const std::string &pattern, const std::string &secure_root);

class Scan
{
public:
  Scan(const Definition &definition, const std::string &secure_root, const std::vector<size_t> &projection);
  ~Scan();
  Scan(const Scan &) = delete;
  Scan &operator=(const Scan &) = delete;
  bool next(std::vector<Cell> &row);
  void rescan();
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Native Parquet reader. Only the selected column pages are decoded.
class ParquetScan
{
public:
  ParquetScan(int fd, uint64_t size, const std::vector<size_t> &projection);
  ~ParquetScan();
  ParquetScan(const ParquetScan &) = delete;
  ParquetScan &operator=(const ParquetScan &) = delete;
  const std::vector<Column> &columns() const;
  uint64_t rows() const;
  bool next(std::vector<Cell> &row);
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class CsvWriter
{
public:
  CsvWriter(const Definition &definition, const std::string &secure_root);
  ~CsvWriter();
  void write(const std::vector<Cell> &row);
  void commit();
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}}} // namespace oceanbase::sql::file_sql
