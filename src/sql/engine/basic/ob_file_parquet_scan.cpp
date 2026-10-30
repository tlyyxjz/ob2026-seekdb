/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "sql/engine/basic/ob_file_scan_utils.h"
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace oceanbase { namespace sql { namespace file_sql {
namespace {
constexpr size_t MAX_PAGE = 8 * 1024 * 1024;
constexpr size_t MAX_MEMORY = 64 * 1024 * 1024;
void check(bool value, const char *message) { if (!value) throw Error(std::string("file_sql: Parquet ") + message); }

struct Node
{
  int64_t integer = 0;
  std::string bytes;
  std::vector<Node> list;
  std::vector<int> field_ids;
  std::vector<Node> fields;
  const Node &get(int id) const
  {
    static const Node empty;
    for (size_t i = 0; i < fields.size(); ++i) if (field_ids[i] == id) return fields[i];
    return empty;
  }
  bool has(int id) const { for (int field : field_ids) if (field == id) return true; return false; }
};

class Compact
{
public:
  Compact(const char *data, size_t size) : data_(data), size_(size) {}
  uint8_t byte() { check(position_ < size_, "truncated metadata"); return static_cast<uint8_t>(data_[position_++]); }
  uint64_t varint()
  {
    uint64_t result = 0;
    for (unsigned shift = 0; shift < 64; shift += 7) {
      uint8_t b = byte();
      check(shift != 63 || (b & 0xfe) == 0, "integer overflow");
      result |= uint64_t(b & 127) << shift;
      if (!(b & 128)) return result;
    }
    throw Error("file_sql: Parquet oversized varint");
  }
  int64_t signed_integer() { uint64_t v = varint(); return static_cast<int64_t>(v >> 1) ^ -static_cast<int64_t>(v & 1); }
  Node value(int type, int depth = 0, bool inline_boolean = false)
  {
    check(depth < 32 && ++nodes_ <= 200000, "metadata exceeds resource limit");
    Node out;
    switch (type) {
      case 1: case 2: out.integer = inline_boolean ? type == 1 : byte() == 1; break;
      case 3: out.integer = static_cast<int8_t>(byte()); break;
      case 4: case 5: case 6: out.integer = signed_integer(); break;
      case 7: for (int i = 0; i < 8; ++i) byte(); break;
      case 8: {
        uint64_t length = varint();
        check(length <= size_ - position_, "truncated metadata string");
        out.bytes.assign(data_ + position_, length); position_ += length;
        break;
      }
      case 9: case 10: {
        uint8_t header = byte(); uint64_t count = header >> 4;
        if (count == 15) count = varint();
        check(count <= 100000, "oversized metadata list");
        for (uint64_t i = 0; i < count; ++i) out.list.push_back(value(header & 15, depth + 1));
        break;
      }
      case 11: {
        uint64_t count = varint(); check(count <= 100000, "oversized metadata map");
        uint8_t kinds = count ? byte() : 0;
        for (uint64_t i = 0; i < count; ++i) { value(kinds >> 4, depth + 1); value(kinds & 15, depth + 1); }
        break;
      }
      case 12: {
        int id = 0;
        for (;;) {
          uint8_t header = byte(); if (header == 0) break;
          id = header >> 4 ? id + (header >> 4) : static_cast<int>(signed_integer());
          check(id > 0 && id <= 32767 && !out.has(id), "invalid or duplicate metadata field");
          out.field_ids.push_back(id);
          out.fields.push_back(value(header & 15, depth + 1, true));
        }
        break;
      }
      default: throw Error("file_sql: Parquet invalid compact-protocol type");
    }
    return out;
  }
  size_t position() const { return position_; }
private:
  const char *data_;
  size_t size_, position_ = 0, nodes_ = 0;
};

uint64_t little(const char *p, size_t size)
{ uint64_t out = 0; for (size_t i = 0; i < size; ++i) out |= uint64_t(static_cast<unsigned char>(p[i])) << (8 * i); return out; }

std::string read_at(int fd, uint64_t offset, size_t size, uint64_t file_size)
{
  check(offset <= file_size && size <= file_size - offset, "read outside file bounds");
  std::string result(size, '\0');
#ifndef _WIN32
  size_t done = 0;
  while (done < size) {
    ssize_t count = ::pread(fd, &result[done], size - done, offset + done);
    if (count < 0 && errno == EINTR) continue;
    check(count > 0, "truncated file or failed read"); done += count;
  }
#else
  throw Error("file_sql: Parquet scanning is not supported on this platform");
#endif
  return result;
}

std::string snappy(const std::string &input, size_t expected)
{
  Compact decoder(input.data(), input.size());
  check(decoder.varint() == expected && expected <= MAX_PAGE, "invalid Snappy output length");
  size_t p = decoder.position();
  std::string output; output.reserve(expected);
  auto number = [&](size_t count) {
    check(count <= input.size() - p, "truncated Snappy copy");
    uint64_t v = little(input.data() + p, count); p += count; return v;
  };
  while (p < input.size() && output.size() < expected) {
    unsigned tag = static_cast<unsigned char>(input[p++]);
    size_t length, offset;
    switch (tag & 3) {
      case 0:
        length = tag >> 2;
        length = length < 60 ? length + 1 : number(length - 59) + 1;
        check(length <= input.size() - p && length <= expected - output.size(), "invalid Snappy literal");
        output.append(input.data() + p, length); p += length; continue;
      case 1: length = 4 + ((tag >> 2) & 7); offset = ((tag & 224) << 3) | number(1); break;
      case 2: length = 1 + (tag >> 2); offset = number(2); break;
      default: length = 1 + (tag >> 2); offset = number(4); break;
    }
    check(offset > 0 && offset <= output.size() && length <= expected - output.size(), "invalid Snappy back-reference");
    while (length--) output.push_back(output[output.size() - offset]);
  }
  check(output.size() == expected && p == input.size(), "Snappy length mismatch");
  return output;
}

std::string decompress(const std::string &input, int codec, size_t expected)
{
  check(expected <= MAX_PAGE, "page exceeds 8 MiB limit");
  if (codec == 0) { check(input.size() == expected, "uncompressed page length mismatch"); return input; }
  if (codec == 1) return snappy(input, expected);
  throw Error("file_sql: Parquet supports UNCOMPRESSED and SNAPPY codecs");
}

std::vector<uint32_t> hybrid(const char *data, size_t size, unsigned bits, size_t count)
{
  check(bits <= 32 && count <= MAX_PAGE, "invalid RLE dimensions");
  Compact decoder(data, size);
  std::vector<uint32_t> values; values.reserve(count);
  while (values.size() < count) {
    uint64_t run = decoder.varint(); check(run != 0, "empty RLE run");
    if (!(run & 1)) {
      size_t n = run >> 1;
      check(n <= count - values.size(), "RLE run exceeds value count");
      uint32_t v = 0;
      for (unsigned i = 0; i < (bits + 7) / 8; ++i) v |= uint32_t(decoder.byte()) << (8 * i);
      check(bits == 32 || v < (uint64_t(1) << bits), "RLE value outside bit width");
      values.insert(values.end(), n, v);
    } else {
      uint64_t groups = run >> 1;
      check(groups <= (count - values.size() + 7) / 8, "bit-packed run exceeds value count");
      uint64_t reservoir = 0; unsigned available = 0;
      for (uint64_t i = 0; i < groups * 8; ++i) {
        while (available < bits) { reservoir |= uint64_t(decoder.byte()) << available; available += 8; }
        uint32_t v = bits ? reservoir & ((uint64_t(1) << bits) - 1) : 0;
        reservoir >>= bits; available -= bits;
        if (values.size() < count) values.push_back(v);
      }
    }
  }
  return values;
}

std::string day_string(int64_t days)
{
  int64_t z = days + 719468;
  int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  unsigned doe = static_cast<unsigned>(z - era * 146097);
  unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  int64_t year = yoe + era * 400;
  unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned mp = (5 * doy + 2) / 153;
  unsigned day = doy - (153 * mp + 2) / 5 + 1;
  unsigned month = mp < 10 ? mp + 3 : mp - 9;
  year += month <= 2;
  check(year >= 1 && year <= 9999, "date outside SQL range");
  std::ostringstream out;
  out << std::setfill('0') << std::setw(4) << year << '-' << std::setw(2) << month << '-' << std::setw(2) << day;
  return out.str();
}

std::string timestamp(int64_t microseconds)
{
  int64_t days = microseconds / 86400000000LL, remaining = microseconds % 86400000000LL;
  if (remaining < 0) { --days; remaining += 86400000000LL; }
  std::ostringstream out;
  out << day_string(days) << ' ' << std::setfill('0') << std::setw(2) << remaining / 3600000000LL << ':'
      << std::setw(2) << remaining / 60000000 % 60 << ':' << std::setw(2) << remaining / 1000000 % 60
      << '.' << std::setw(6) << remaining % 1000000;
  return out.str();
}

struct Leaf
{
  Column column;
  int physical = 0, width = 0, converted = -1, timestamp_unit = 0;
  bool optional = false, unsigned_integer = false;
};

std::vector<Cell> plain(const std::string &data, size_t start, size_t count, const Leaf &leaf)
{
  check(count <= 262144, "page exceeds value limit");
  std::vector<Cell> result; result.reserve(count);
  size_t p = start;
  auto number = [&](size_t width) {
    check(width <= data.size() - p, "truncated plain value");
    uint64_t v = little(data.data() + p, width); p += width; return v;
  };
  for (size_t i = 0; i < count; ++i) {
    Cell cell; cell.is_null = false; cell.type = leaf.column.type;
    switch (leaf.physical) {
      case 0:
        check(start + i / 8 < data.size(), "truncated boolean page");
        cell.text = (static_cast<unsigned char>(data[start + i / 8]) >> (i % 8)) & 1 ? "true" : "false";
        break;
      case 1: {
        uint32_t raw = number(4);
        int64_t value = leaf.unsigned_integer ? int64_t(raw) : int64_t(static_cast<int32_t>(raw));
        cell.text = leaf.column.type == Type::Date ? day_string(value) : std::to_string(value);
        break;
      }
      case 2: {
        uint64_t raw = number(8);
        if (leaf.unsigned_integer) {
          check(raw <= uint64_t(std::numeric_limits<int64_t>::max()), "unsigned integer exceeds BIGINT");
          cell.text = std::to_string(raw);
        } else if (leaf.column.type == Type::Datetime) {
          int64_t value = static_cast<int64_t>(raw);
          if (leaf.timestamp_unit == 1) {
            check(value <= std::numeric_limits<int64_t>::max() / 1000 && value >= std::numeric_limits<int64_t>::min() / 1000, "timestamp overflow");
            value *= 1000;
          } else if (leaf.timestamp_unit == 3) value /= 1000;
          cell.text = timestamp(value);
        } else cell.text = std::to_string(static_cast<int64_t>(raw));
        break;
      }
      case 3: {
        uint64_t nanos = number(8), julian = number(4);
        check(nanos < 86400000000000ULL && julian >= 1721426 && julian <= 5373484, "invalid INT96 timestamp");
        cell.text = timestamp((static_cast<int64_t>(julian) - 2440588) * 86400000000LL + nanos / 1000);
        break;
      }
      case 4: case 5: {
        double value;
        if (leaf.physical == 4) { uint32_t raw = number(4); float f; std::memcpy(&f, &raw, 4); value = f; }
        else { uint64_t raw = number(8); std::memcpy(&value, &raw, 8); }
        check(std::isfinite(value), "non-finite floating-point value");
        std::ostringstream out; out << std::setprecision(17) << value; cell.text = out.str(); break;
      }
      case 6: case 7: {
        uint64_t length = leaf.physical == 6 ? number(4) : leaf.width;
        check(length <= 4 * 1024 * 1024 && length <= data.size() - p, "invalid string length");
        cell.text.assign(data.data() + p, length); p += length;
        check(valid_utf8(cell.text), "binary values are not valid UTF-8 VARCHAR data"); break;
      }
      default: throw Error("file_sql: unsupported Parquet physical type");
    }
    result.push_back(std::move(cell));
  }
  return result;
}

struct Chunk { uint64_t start = 0, end = 0; int codec = 0; };
struct Group { uint64_t rows = 0; std::vector<Chunk> chunks; };

class ColumnReader
{
public:
  ColumnReader(int fd, uint64_t size, const Leaf &leaf, const Chunk &chunk)
    : fd_(fd), size_(size), leaf_(leaf), chunk_(chunk), offset_(chunk.start) {}
  Cell next()
  {
    while (index_ == values_.size()) read_page();
    return values_[index_++];
  }
  size_t memory() const { return memory_; }
private:
  void read_page()
  {
    check(offset_ < chunk_.end, "column ended before row group");
    std::string header_bytes = read_at(fd_, offset_, std::min<uint64_t>(65536, chunk_.end - offset_), size_);
    Compact protocol(header_bytes.data(), header_bytes.size());
    Node header = protocol.value(12);
    int64_t compressed = header.get(3).integer, uncompressed = header.get(2).integer;
    check(header.has(1) && header.has(2) && header.has(3) && compressed >= 0 && uncompressed >= 0
          && compressed <= int64_t(MAX_PAGE) && uncompressed <= int64_t(MAX_PAGE), "invalid page header");
    offset_ += protocol.position();
    check(uint64_t(compressed) <= chunk_.end - offset_, "page exceeds column chunk");
    std::string encoded = read_at(fd_, offset_, compressed, size_); offset_ += compressed;
    int type = header.get(1).integer;
    if (type != 0 && type != 2 && type != 3) { values_.clear(); index_ = 0; return; }
    values_.clear(); index_ = 0;
    size_t count, encoding, start = 0;
    std::vector<uint32_t> levels;
    std::string decoded;
    if (type == 2) {
      const Node &dictionary = header.get(7);
      check(dictionary.has(1) && dictionary.get(1).integer >= 0 && dictionary.get(2).integer == 0, "invalid dictionary page");
      decoded = decompress(encoded, chunk_.codec, uncompressed);
      dictionary_ = plain(decoded, 0, dictionary.get(1).integer, leaf_);
      update_memory(); return;
    } else if (type == 0) {
      const Node &page = header.get(5);
      check(page.has(1) && page.get(1).integer >= 0 && page.get(1).integer <= 262144, "invalid data page value count");
      count = page.get(1).integer; encoding = page.get(2).integer;
      decoded = decompress(encoded, chunk_.codec, uncompressed);
      if (leaf_.optional) {
        check(page.get(3).integer == 3 && decoded.size() >= 4, "unsupported definition-level encoding");
        uint64_t bytes = little(decoded.data(), 4);
        check(bytes <= decoded.size() - 4, "invalid definition-level length");
        levels = hybrid(decoded.data() + 4, bytes, 1, count); start = 4 + bytes;
      }
    } else {
      const Node &page = header.get(8);
      check(page.has(1) && page.get(1).integer >= 0 && page.get(1).integer <= 262144, "invalid v2 value count");
      count = page.get(1).integer; encoding = page.get(4).integer;
      int64_t def_size = page.get(5).integer, rep_size = page.get(6).integer;
      check(def_size >= 0 && rep_size == 0 && uint64_t(def_size) <= encoded.size() && def_size <= uncompressed, "invalid v2 levels");
      if (leaf_.optional) levels = hybrid(encoded.data(), def_size, 1, count);
      decoded = decompress(encoded.substr(def_size), page.has(7) && !page.get(7).integer ? 0 : chunk_.codec, uncompressed - def_size);
    }
    if (!leaf_.optional) levels.assign(count, 1);
    size_t nonnull = std::count(levels.begin(), levels.end(), 1);
    std::vector<Cell> present;
    if (encoding == 0) present = plain(decoded, start, nonnull, leaf_);
    else if (encoding == 3 && leaf_.physical == 0) {
      // Boolean RLE retains its four-byte length prefix in both page versions.
      // https://parquet.apache.org/docs/file-format/data-pages/encodings/
      check(start <= decoded.size() && decoded.size() - start >= 4, "truncated boolean RLE length");
      uint64_t length = little(decoded.data() + start, 4); start += 4;
      check(length <= decoded.size() - start, "invalid boolean RLE length");
      for (uint32_t value : hybrid(decoded.data() + start, length, 1, nonnull))
        present.push_back({false, value ? "true" : "false", Type::Boolean});
    }
    else if (encoding == 2 || encoding == 8) {
      check(start < decoded.size() || nonnull == 0, "truncated dictionary indices");
      if (nonnull) {
        unsigned width = static_cast<unsigned char>(decoded[start++]);
        auto indices = hybrid(decoded.data() + start, decoded.size() - start, width, nonnull);
        for (uint32_t i : indices) { check(i < dictionary_.size(), "dictionary index out of range"); present.push_back(dictionary_[i]); }
      }
    } else throw Error("file_sql: unsupported Parquet data encoding");
    size_t position = 0;
    for (uint32_t level : levels) { check(level <= 1, "invalid null definition level"); values_.push_back(level ? std::move(present[position++]) : Cell{}); }
    update_memory();
  }
  void update_memory()
  {
    memory_ = (values_.capacity() + dictionary_.capacity()) * sizeof(Cell);
    for (const auto &c : values_) memory_ += c.text.capacity();
    for (const auto &c : dictionary_) memory_ += c.text.capacity();
    check(memory_ <= MAX_MEMORY, "column page exceeds memory budget");
  }
  int fd_;
  uint64_t size_;
  Leaf leaf_;
  Chunk chunk_;
  uint64_t offset_;
  size_t index_ = 0, memory_ = 0;
  std::vector<Cell> dictionary_, values_;
};
} // namespace

class ParquetScan::Impl
{
public:
  Impl(int fd, uint64_t size, const std::vector<size_t> &projection) : fd_(fd), size_(size), projection_(projection)
  {
    check(size >= 12 && read_at(fd, 0, 4, size) == "PAR1", "missing file magic");
    std::string tail = read_at(fd, size - 8, 8, size);
    check(tail.substr(4) == "PAR1", "missing footer magic");
    uint64_t footer_size = little(tail.data(), 4);
    check(footer_size <= 16 * 1024 * 1024 && footer_size <= size - 12, "invalid footer size");
    uint64_t footer_offset = size - 8 - footer_size;
    std::string footer = read_at(fd, footer_offset, footer_size, size);
    Compact protocol(footer.data(), footer.size()); Node metadata = protocol.value(12);
    const auto &schema = metadata.get(2).list;
    check(!schema.empty() && schema.size() <= 1025 && schema[0].get(5).integer == int64_t(schema.size() - 1), "only flat schemas are supported");
    for (size_t i = 1; i < schema.size(); ++i) {
      const Node &s = schema[i];
      check(s.has(1) && !s.has(5) && s.get(3).integer <= 1, "nested and repeated columns are not supported");
      Leaf leaf;
      leaf.physical = s.get(1).integer; leaf.optional = s.get(3).integer == 1;
      leaf.width = s.get(2).integer; leaf.converted = s.has(6) ? s.get(6).integer : -1;
      leaf.column.name = s.get(4).bytes; leaf.column.nullable = leaf.optional;
      check(!leaf.column.name.empty() && valid_utf8(leaf.column.name), "invalid column name");
      switch (leaf.physical) {
        case 0: leaf.column.type = Type::Boolean; break;
        case 1: case 2: leaf.column.type = Type::Bigint; break;
        case 3: leaf.column.type = Type::Datetime; break;
        case 4: case 5: leaf.column.type = Type::Double; break;
        case 6: case 7: leaf.column.type = Type::Varchar; break;
        default: throw Error("file_sql: unsupported Parquet schema type");
      }
      if (leaf.physical == 7) check(leaf.width > 0 && leaf.width <= 4 * 1024 * 1024, "invalid fixed-width column");
      if (leaf.converted == 6 || s.get(10).has(6)) { check(leaf.physical == 1, "invalid DATE physical type"); leaf.column.type = Type::Date; }
      if (leaf.converted == 9 || leaf.converted == 10 || s.get(10).has(8)) {
        check(leaf.physical == 2, "invalid TIMESTAMP physical type"); leaf.column.type = Type::Datetime;
        leaf.timestamp_unit = leaf.converted == 9 ? 1 : 2;
        if (s.get(10).has(8)) {
          const Node &unit = s.get(10).get(8).get(2);
          leaf.timestamp_unit = unit.has(1) ? 1 : unit.has(2) ? 2 : 3;
        }
      }
      leaf.unsigned_integer = leaf.converted >= 11 && leaf.converted <= 14;
      if (leaf.converted == 25 || s.get(10).has(11)) leaf.column.type = Type::Null;
      if (s.get(10).has(10)) leaf.unsigned_integer = !s.get(10).get(10).get(2).integer;
      check(leaf.converted != 5 && !s.get(10).has(5), "decimal columns are not supported");
      if (leaf.column.type == Type::Datetime) leaf.column.scale = leaf.timestamp_unit == 1 ? 3 : 6;
      leaves_.push_back(leaf); columns_.push_back(leaf.column);
    }
    check(!leaves_.empty() && metadata.get(3).integer >= 0, "invalid row count");
    rows_ = metadata.get(3).integer;
    uint64_t count = 0;
    for (const Node &row_group : metadata.get(4).list) {
      Group group;
      check(row_group.get(3).integer >= 0 && row_group.get(1).list.size() == leaves_.size(), "invalid row group");
      group.rows = row_group.get(3).integer;
      for (size_t i = 0; i < leaves_.size(); ++i) {
        const Node &chunk = row_group.get(1).list[i], &m = chunk.get(3);
        check(!chunk.has(1) || chunk.get(1).bytes.empty(), "external column files are not supported");
        check(m.get(3).list.size() == 1 && m.get(3).list[0].bytes == leaves_[i].column.name
              && m.get(1).integer == leaves_[i].physical && m.get(5).integer == int64_t(group.rows), "column metadata mismatch");
        check((m.get(9).integer >= 4 || (group.rows == 0 && m.get(9).integer == 0))
              && m.get(7).integer >= 0, "invalid column offset");
        Chunk c;
        c.start = m.get(9).integer == 0 ? (m.has(11) ? m.get(11).integer : 4)
                  : m.has(11) ? std::min(m.get(11).integer, m.get(9).integer) : m.get(9).integer;
        c.codec = m.get(4).integer;
        check(c.start >= 4 && c.start <= footer_offset && uint64_t(m.get(7).integer) <= footer_offset - c.start, "column outside data section");
        c.end = c.start + m.get(7).integer;
        group.chunks.push_back(c);
      }
      check(group.rows <= std::numeric_limits<uint64_t>::max() - count, "row count overflow");
      count += group.rows; groups_.push_back(std::move(group));
    }
    check(count == rows_, "row group count mismatch");
    for (size_t i : projection_) check(i < leaves_.size(), "invalid projected column");
  }
  bool next(std::vector<Cell> &row)
  {
    while (group_ < groups_.size() && row_ == groups_[group_].rows) { ++group_; row_ = 0; readers_.clear(); }
    if (group_ == groups_.size()) return false;
    if (readers_.empty() && !projection_.empty()) {
      for (size_t i : projection_) readers_.emplace_back(new ColumnReader(fd_, size_, leaves_[i], groups_[group_].chunks[i]));
    }
    row.clear(); size_t memory = 0;
    for (auto &reader : readers_) { row.push_back(reader->next()); memory += reader->memory(); }
    check(memory <= MAX_MEMORY, "projected pages exceed 64 MiB memory budget");
    ++row_; return true;
  }
  int fd_;
  uint64_t size_, rows_ = 0;
  std::vector<size_t> projection_;
  std::vector<Leaf> leaves_;
  std::vector<Column> columns_;
  std::vector<Group> groups_;
  size_t group_ = 0;
  uint64_t row_ = 0;
  std::vector<std::unique_ptr<ColumnReader>> readers_;
};

ParquetScan::ParquetScan(int fd, uint64_t size, const std::vector<size_t> &projection) : impl_(new Impl(fd, size, projection)) {}
ParquetScan::~ParquetScan() = default;
const std::vector<Column> &ParquetScan::columns() const { return impl_->columns_; }
uint64_t ParquetScan::rows() const { return impl_->rows_; }
bool ParquetScan::next(std::vector<Cell> &row) { return impl_->next(row); }

}}} // namespace oceanbase::sql::file_sql
