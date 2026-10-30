/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "sql/engine/basic/ob_file_scan_utils.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <unordered_map>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#ifndef _WIN32
#include <fcntl.h>
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/inotify.h>
#endif
#else
#include <io.h>
#define read _read
#define close _close
#define lseek _lseeki64
#endif

namespace oceanbase { namespace sql { namespace file_sql {
namespace {
constexpr size_t MAX_ROW = 4 * 1024 * 1024;
constexpr size_t MAX_COLUMNS = 1024;
constexpr char MAGIC[] = "SEEKDB_FILE_SQL_1:";

struct Fd
{
  int value = -1;
  Fd() = default;
  explicit Fd(int v) : value(v) {}
  ~Fd() { if (value >= 0) ::close(value); }
  Fd(const Fd &) = delete;
  Fd &operator=(const Fd &) = delete;
};

// Timestamps can be coalesced for same-size writes. Check the contents when
// opening a scan, and watch writes while consuming it, without retaining data.
uint64_t hash_file(int fd)
{
#ifndef _WIN32
  char buffer[64 * 1024];
  uint64_t hash = 14695981039346656037ULL;
  off_t offset = 0;
  for (;;) {
    ssize_t count = ::pread(fd, buffer, sizeof(buffer), offset);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) throw Error("file_sql: cannot fingerprint input");
    if (count == 0) return hash;
    for (ssize_t i = 0; i < count; ++i) { hash ^= static_cast<unsigned char>(buffer[i]); hash *= 1099511628211ULL; }
    offset += count;
  }
#else
  throw Error("file_sql: local file scanning is not supported on this platform");
#endif
}

class ChangeWatch
{
public:
  void reset(const std::string &path)
  {
#ifdef __linux__
    if (fd_.value >= 0) ::close(fd_.value);
    fd_.value = ::inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (fd_.value < 0 || ::inotify_add_watch(fd_.value, path.c_str(),
        IN_MODIFY | IN_ATTRIB | IN_MOVE_SELF | IN_DELETE_SELF | IN_UNMOUNT | IN_CLOSE_WRITE) < 0)
      throw Error("file_sql: cannot watch input changes");
#endif
  }
  void check()
  {
#ifdef __linux__
    char events[4096];
    ssize_t size;
    do { size = ::read(fd_.value, events, sizeof(events)); } while (size < 0 && errno == EINTR);
    if (size > 0) throw Error("file_sql: input changed during scan");
    if (size < 0 && errno != EAGAIN) throw Error("file_sql: cannot check input changes");
#endif
  }
private:
  Fd fd_;
};

std::string lower(std::string s)
{
  for (char &c : s) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
  return s;
}

void require(bool condition, const std::string &message)
{
  if (!condition) throw Error(message);
}

#ifndef _WIN32
Stamp stamp_fd(int fd)
{
  struct stat st;
  require(::fstat(fd, &st) == 0 && S_ISREG(st.st_mode), "file_sql: input must be a readable regular file");
  Stamp result;
  result.device = st.st_dev; result.inode = st.st_ino; result.size = st.st_size;
#ifdef __APPLE__
  result.mtime = st.st_mtimespec.tv_sec; result.mtime_ns = st.st_mtimespec.tv_nsec;
  result.ctime = st.st_ctimespec.tv_sec; result.ctime_ns = st.st_ctimespec.tv_nsec;
#else
  result.mtime = st.st_mtim.tv_sec; result.mtime_ns = st.st_mtim.tv_nsec;
  result.ctime = st.st_ctim.tv_sec; result.ctime_ns = st.st_ctim.tv_nsec;
#endif
  return result;
}

std::string canonical(const std::string &path)
{
  require(!path.empty() && path.find('\0') == std::string::npos && path.find("://") == std::string::npos,
          "file_sql: expected a local path without NUL bytes");
  char *resolved = ::realpath(path.c_str(), nullptr);
  require(resolved != nullptr, "file_sql: cannot resolve path: " + path);
  std::string result(resolved);
  ::free(resolved);
  return result;
}

int secure_open(const std::string &path, const std::string &root, std::string &resolved, bool is_directory = false)
{
  require(!root.empty() && lower(root) != "null", "file_sql: secure_file_priv disables file access");
  resolved = canonical(path);
  std::string base = root.empty() ? "/" : canonical(root);
  require(base == "/" || (is_directory && resolved == base) || (resolved.size() > base.size() && resolved.compare(0, base.size(), base) == 0
          && resolved[base.size()] == '/'), "file_sql: path is outside secure_file_priv: " + path);
  Fd directory(::open(base.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  require(directory.value >= 0, "file_sql: cannot open secure directory");
  if (resolved == base && is_directory) { int fd = directory.value; directory.value = -1; return fd; }
  std::string relative = resolved.substr(base == "/" ? 1 : base.size() + 1);
  size_t start = 0;
  for (;;) {
    size_t slash = relative.find('/', start);
    std::string part = relative.substr(start, slash - start);
    require(!part.empty() && part != "." && part != "..", "file_sql: invalid path component");
    int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (slash == std::string::npos && !is_directory ? O_NONBLOCK : O_DIRECTORY);
    int next = ::openat(directory.value, part.c_str(), flags);
    require(next >= 0, "file_sql: cannot securely open path: " + path);
    if (slash == std::string::npos) {
      Fd file(next);
      if (!is_directory) stamp_fd(file.value);
      file.value = -1;
      return next;
    }
    ::close(directory.value);
    directory.value = next;
    start = slash + 1;
  }
}
#else
Stamp stamp_fd(int) { throw Error("file_sql: local file scanning is not supported on this platform"); }
int secure_open(const std::string &, const std::string &, std::string &, bool = false)
{ throw Error("file_sql: local file scanning is not supported on this platform"); }
#endif

bool date(const std::string &s)
{
  if (s.size() != 10 || s[4] != '-' || s[7] != '-') return false;
  for (size_t i = 0; i < s.size(); ++i) if (i != 4 && i != 7 && (s[i] < '0' || s[i] > '9')) return false;
  int y = std::stoi(s.substr(0, 4)), m = std::stoi(s.substr(5, 2)), d = std::stoi(s.substr(8, 2));
  static const int days[] = {0,31,28,31,30,31,30,31,31,30,31,30,31};
  return y >= 1 && m >= 1 && m <= 12 && d >= 1
      && d <= days[m] + (m == 2 && (y % 400 == 0 || (y % 4 == 0 && y % 100 != 0)));
}

bool datetime(const std::string &s)
{
  if (s.size() < 19 || (s[10] != ' ' && s[10] != 'T') || s[13] != ':' || s[16] != ':' || !date(s.substr(0, 10))) return false;
  for (size_t i : {11U,12U,14U,15U,17U,18U}) if (s[i] < '0' || s[i] > '9') return false;
  if (std::stoi(s.substr(11,2)) > 23 || std::stoi(s.substr(14,2)) > 59 || std::stoi(s.substr(17,2)) > 59) return false;
  if (s.size() == 19) return true;
  if (s.size() < 21 || s.size() > 26 || s[19] != '.') return false;
  return std::all_of(s.begin() + 20, s.end(), [](char c) { return c >= '0' && c <= '9'; });
}

class Input
{
public:
  explicit Input(int fd) : fd_(fd) {}
  int get()
  {
    if (pos_ == size_) {
      int64_t count;
      do { count = ::read(fd_, buffer_, sizeof(buffer_)); } while (count < 0 && errno == EINTR);
      require(count >= 0, "file_sql: read failed");
      pos_ = 0; size_ = count;
      if (size_ == 0) return -1;
      if (first_) {
        first_ = false;
        if (size_ >= 3 && std::memcmp(buffer_, "\xef\xbb\xbf", 3) == 0) pos_ = 3;
        if (pos_ == size_) return get();
      }
    }
    return static_cast<unsigned char>(buffer_[pos_++]);
  }
  void reset() { require(::lseek(fd_, 0, SEEK_SET) == 0, "file_sql: cannot rewind file"); pos_ = size_ = 0; line_ = 1; first_ = true; }
  uint64_t line_ = 1;
private:
  int fd_;
  char buffer_[64 * 1024];
  size_t pos_ = 0, size_ = 0;
  bool first_ = true;
};

bool csv_row(Input &in, const Options &options, std::vector<Cell> &row)
{
  for (;;) {
    row.clear();
    std::string value;
    bool quoted = false, in_quote = false, closed = false, started = false;
    size_t bytes = 0;
    auto finish = [&]() {
      require(row.size() < MAX_COLUMNS, "file_sql: too many CSV columns");
      require(valid_utf8(value), "file_sql: invalid UTF-8 at CSV line " + std::to_string(in.line_));
      Cell cell;
      cell.is_null = !quoted && (value.empty() || value == options.null_string);
      cell.text.swap(value);
      cell.type = cell.is_null ? Type::Null : infer_type(cell.text);
      row.push_back(std::move(cell));
      quoted = in_quote = closed = started = false;
    };
    int c;
    while ((c = in.get()) >= 0) {
      require(++bytes <= MAX_ROW, "file_sql: CSV record exceeds 4 MiB at line " + std::to_string(in.line_));
      if (c == '\r') { int next = in.get(); require(next == '\n' || next == -1, "file_sql: bare CR in CSV"); c = '\n'; }
      if (c == '\n') ++in.line_;
      if (in_quote) {
        if (options.escape && options.escape != options.quote && c == options.escape) {
          int next = in.get();
          require(next >= 0, "file_sql: truncated CSV escape");
          value.push_back(static_cast<char>(next));
        } else if (c == options.quote) {
          in_quote = false; closed = true;
        } else {
          value.push_back(static_cast<char>(c));
        }
      } else if (closed && c == options.quote && options.escape == options.quote) {
        value.push_back(static_cast<char>(c)); in_quote = true; closed = false;
      } else if (c == options.delimiter) {
        finish();
      } else if (c == '\n') {
        bool blank = bytes <= 2 && row.empty() && value.empty() && !quoted;
        finish();
        if (!blank) return true;
        break;
      } else if (closed) {
        require(c == ' ' || c == '\t', "file_sql: characters after CSV quote at line " + std::to_string(in.line_));
      } else if (!started && options.quote && c == options.quote) {
        quoted = in_quote = started = true;
      } else {
        require(!options.quote || c != options.quote, "file_sql: unexpected CSV quote at line " + std::to_string(in.line_));
        started = true; value.push_back(static_cast<char>(c));
      }
    }
    if (c < 0) {
      require(!in_quote, "file_sql: unterminated CSV quote at line " + std::to_string(in.line_));
      if (bytes == 0 && row.empty()) return false;
      finish();
      return true;
    }
  }
}

using ObjectRow = std::vector<std::pair<std::string, Cell>>;
bool json_row(Input &input, ObjectRow &row)
{
  std::string line;
  for (;;) {
    line.clear();
    int c;
    while ((c = input.get()) >= 0 && c != '\n') {
      require(line.size() < MAX_ROW, "file_sql: JSONL record exceeds 4 MiB");
      line.push_back(static_cast<char>(c));
    }
    uint64_t number = input.line_++;
    if (line.find_first_not_of(" \t\r") == std::string::npos) { if (c < 0) return false; else continue; }
    rapidjson::Document document;
    document.Parse<rapidjson::kParseValidateEncodingFlag | rapidjson::kParseFullPrecisionFlag>(line.data(), line.size());
    std::string where = " at JSONL line " + std::to_string(number);
    require(!document.HasParseError() && document.IsObject(), "file_sql: expected a flat JSON object" + where);
    row.clear();
    std::set<std::string> names;
    for (auto member = document.MemberBegin(); member != document.MemberEnd(); ++member) {
      require(row.size() < MAX_COLUMNS, "file_sql: too many JSON fields" + where);
      std::string name(member->name.GetString(), member->name.GetStringLength());
      require(!name.empty() && names.insert(lower(name)).second, "file_sql: duplicate or empty JSON field" + where);
      const auto &v = member->value;
      Cell cell;
      cell.is_null = v.IsNull();
      if (v.IsString()) {
        cell.text.assign(v.GetString(), v.GetStringLength());
        Type inferred = infer_type(cell.text);
        cell.type = inferred == Type::Date || inferred == Type::Datetime ? inferred : Type::Varchar;
      } else if (v.IsBool()) {
        cell.text = v.GetBool() ? "true" : "false"; cell.type = Type::Boolean;
      } else if (v.IsNumber()) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        v.Accept(writer);
        cell.text.assign(buffer.GetString(), buffer.GetSize());
        cell.type = infer_type(cell.text);
      } else require(v.IsNull(), "file_sql: nested JSON objects and arrays are not supported" + where);
      row.emplace_back(std::move(name), std::move(cell));
    }
    return true;
  }
}

void unique_names(std::vector<Column> &columns)
{
  std::set<std::string> used;
  for (size_t i = 0; i < columns.size(); ++i) {
    std::string base = columns[i].name.empty() ? "column" + std::to_string(i) : columns[i].name;
    require(base.size() <= 256 && base.find('\0') == std::string::npos, "file_sql: invalid or excessive column name");
    std::string name = base;
    size_t count = 1;
    while (!used.insert(lower(name)).second) name = base + "__" + std::to_string(++count);
    columns[i].name = std::move(name);
  }
}

void write_integer(std::string &out, uint64_t value)
{ for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>(value >> (i * 8))); }
void write_string(std::string &out, const std::string &value)
{ write_integer(out, value.size()); out.append(value); }
uint64_t read_integer(const std::string &in, size_t &offset)
{
  require(offset <= in.size() && in.size() - offset >= 8, "file_sql: invalid plan descriptor");
  uint64_t result = 0;
  for (int i = 0; i < 8; ++i) result |= static_cast<uint64_t>(static_cast<unsigned char>(in[offset++])) << (i * 8);
  return result;
}
std::string read_string(const std::string &in, size_t &offset)
{
  uint64_t size = read_integer(in, offset);
  require(size <= MAX_ROW && size <= in.size() - offset, "file_sql: invalid descriptor string");
  std::string result = in.substr(offset, size); offset += size; return result;
}
} // namespace

bool Stamp::operator==(const Stamp &o) const
{
  return device == o.device && inode == o.inode && size == o.size && mtime == o.mtime
      && mtime_ns == o.mtime_ns && ctime == o.ctime && ctime_ns == o.ctime_ns;
}

const char *type_name(Type type)
{
  static const char *names[] = {"NULL", "BOOLEAN", "BIGINT", "DOUBLE", "VARCHAR", "DATE", "DATETIME"};
  require(static_cast<unsigned>(type) < 7, "file_sql: invalid column type");
  return names[static_cast<unsigned>(type)];
}

bool valid_utf8(const std::string &s)
{
  for (size_t i = 0; i < s.size();) {
    unsigned char a = s[i++];
    if (a < 0x80) continue;
    unsigned n = a >= 0xc2 && a <= 0xdf ? 1 : a >= 0xe0 && a <= 0xef ? 2 : a >= 0xf0 && a <= 0xf4 ? 3 : 0;
    if (!n || s.size() - i < n) return false;
    unsigned char b = s[i];
    if ((a == 0xe0 && b < 0xa0) || (a == 0xed && b >= 0xa0) || (a == 0xf0 && b < 0x90) || (a == 0xf4 && b >= 0x90)) return false;
    while (n--) if ((static_cast<unsigned char>(s[i++]) & 0xc0) != 0x80) return false;
  }
  return true;
}

Type infer_type(const std::string &text)
{
  if (text.empty()) return Type::Varchar;
  std::string folded = lower(text);
  if (folded == "true" || folded == "false") return Type::Boolean;
  if (date(text)) return Type::Date;
  if (datetime(text)) return Type::Datetime;
  const char *begin = text.data(), *end = begin + text.size();
  if (*begin == '+' && begin + 1 < end) ++begin;
  int64_t integer = 0;
  auto parsed = std::from_chars(begin, end, integer);
  if (parsed.ptr == end && parsed.ec == std::errc()) return Type::Bigint;
  bool integer_shape = std::all_of(begin + (*begin == '-' ? 1 : 0), end, [](char c) { return c >= '0' && c <= '9'; });
  if (integer_shape) return Type::Varchar; // Keep overflowing integers exact.
  double value = 0;
  auto floating = std::from_chars(begin, end, value, std::chars_format::general);
  return floating.ptr == end && floating.ec == std::errc() && std::isfinite(value) ? Type::Double : Type::Varchar;
}

Type widen(Type a, Type b)
{
  if (a == Type::Null) return b;
  if (b == Type::Null || a == b) return a;
  auto numeric = [](Type t) { return t == Type::Boolean || t == Type::Bigint || t == Type::Double; };
  if (numeric(a) && numeric(b)) return a == Type::Double || b == Type::Double ? Type::Double : Type::Bigint;
  if ((a == Type::Date && b == Type::Datetime) || (b == Type::Date && a == Type::Datetime)) return Type::Datetime;
  return Type::Varchar;
}

bool Definition::is_descriptor(const char *data, int64_t length)
{ return length >= static_cast<int64_t>(sizeof(MAGIC) - 1) && std::memcmp(data, MAGIC, sizeof(MAGIC) - 1) == 0; }

std::string Definition::serialize() const
{
  std::string out(MAGIC);
  write_string(out, path);
  write_string(out, source_path);
  write_integer(out, content_hash);
  for (uint64_t v : {static_cast<uint64_t>(options.format), static_cast<uint64_t>(options.header),
                    static_cast<uint64_t>(static_cast<unsigned char>(options.delimiter)),
                    static_cast<uint64_t>(static_cast<unsigned char>(options.quote)),
                    static_cast<uint64_t>(static_cast<unsigned char>(options.escape))}) write_integer(out, v);
  write_string(out, options.null_string);
  for (uint64_t v : {stamp.device, stamp.inode, stamp.size, static_cast<uint64_t>(stamp.mtime), static_cast<uint64_t>(stamp.mtime_ns),
                    static_cast<uint64_t>(stamp.ctime), static_cast<uint64_t>(stamp.ctime_ns), rows}) write_integer(out, v);
  write_integer(out, columns.size());
  for (const auto &c : columns) {
    write_string(out, c.name); write_integer(out, static_cast<uint64_t>(c.type));
    write_integer(out, c.nullable); write_integer(out, c.scale);
  }
  return out;
}

Definition Definition::deserialize(const std::string &bytes)
{
  require(is_descriptor(bytes.data(), bytes.size()) && bytes.size() <= MAX_ROW, "file_sql: invalid descriptor");
  Definition d;
  size_t p = sizeof(MAGIC) - 1;
  d.path = read_string(bytes, p);
  d.source_path = read_string(bytes, p);
  d.content_hash = read_integer(bytes, p);
  uint64_t format = read_integer(bytes, p);
  require(format <= static_cast<uint64_t>(Format::Glob), "file_sql: invalid format");
  d.options.format = static_cast<Format>(format);
  d.options.header = read_integer(bytes, p) != 0;
  d.options.delimiter = read_integer(bytes, p); d.options.quote = read_integer(bytes, p); d.options.escape = read_integer(bytes, p);
  d.options.null_string = read_string(bytes, p);
  d.stamp.device = read_integer(bytes, p); d.stamp.inode = read_integer(bytes, p); d.stamp.size = read_integer(bytes, p);
  d.stamp.mtime = read_integer(bytes, p); d.stamp.mtime_ns = read_integer(bytes, p);
  d.stamp.ctime = read_integer(bytes, p); d.stamp.ctime_ns = read_integer(bytes, p); d.rows = read_integer(bytes, p);
  uint64_t count = read_integer(bytes, p);
  require(count > 0 && count <= MAX_COLUMNS, "file_sql: invalid column count");
  for (uint64_t i = 0; i < count; ++i) {
    Column column;
    column.name = read_string(bytes, p);
    uint64_t type = read_integer(bytes, p);
    require(type <= static_cast<uint64_t>(Type::Datetime), "file_sql: invalid type");
    column.type = static_cast<Type>(type); column.nullable = read_integer(bytes, p) != 0;
    uint64_t scale = read_integer(bytes, p);
    require(scale <= 6, "file_sql: invalid datetime scale"); column.scale = scale;
    d.columns.push_back(std::move(column));
  }
  require(p == bytes.size(), "file_sql: trailing descriptor data");
  return d;
}

std::vector<std::string> discover(const std::string &pattern, const std::string &secure_root)
{
#ifndef _WIN32
  require(!secure_root.empty() && lower(secure_root) != "null" && pattern.find('\0') == std::string::npos,
          "file_sql: invalid glob or disabled file access");
  size_t magic = pattern.find_first_of("*?[");
  size_t slash = pattern.rfind('/', magic == std::string::npos ? pattern.size() : magic);
  std::string parent = slash == std::string::npos ? "." : pattern.substr(0, slash + 1);
  std::string base = secure_root.empty() ? "/" : canonical(secure_root);
  std::string resolved_parent = canonical(parent);
  require(base == "/" || resolved_parent == base || (resolved_parent.compare(0, base.size(), base) == 0
          && resolved_parent.size() > base.size() && resolved_parent[base.size()] == '/'), "file_sql: glob is outside secure_file_priv");
  glob_t matches{};
  int status = ::glob(pattern.c_str(), GLOB_NOSORT, nullptr, &matches);
  std::vector<std::string> result;
  try {
    require(status == 0 || status == GLOB_NOMATCH, "file_sql: glob failed");
    require(matches.gl_pathc <= 10000, "file_sql: glob exceeds 10000 files");
    for (size_t i = 0; i < matches.gl_pathc; ++i) {
      struct stat st;
      require(::stat(matches.gl_pathv[i], &st) == 0, "file_sql: glob entry changed");
      if (S_ISDIR(st.st_mode)) continue;
      std::string resolved;
      Fd fd(secure_open(matches.gl_pathv[i], secure_root, resolved));
      result.push_back(resolved);
    }
  } catch (...) { ::globfree(&matches); throw; }
  ::globfree(&matches);
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
#else
  throw Error("file_sql: glob is not supported on this platform");
#endif
}

Definition infer(const std::string &path, const std::string &secure_root, const Options &options)
{
  require(options.sample_size == -1, "file_sql: only full schema inference (sample_size=-1) is supported");
  require(options.delimiter && options.delimiter != '\n' && options.delimiter != '\r'
          && (!options.quote || options.quote != options.delimiter), "file_sql: invalid CSV delimiter or quote");
  Definition result;
  result.options = options;
  result.source_path = path;
  if (options.format == Format::Glob) {
    result.path = path; result.columns.push_back({"file", Type::Varchar, false});
    result.rows = discover(path, secure_root).size();
    return result;
  }
  Fd fd(secure_open(path, secure_root, result.path));
  ChangeWatch watch;
  watch.reset(result.path);
  result.stamp = stamp_fd(fd.value);
  if (options.format == Format::Parquet) {
    ParquetScan parquet(fd.value, result.stamp.size, {});
    result.columns = parquet.columns(); result.rows = parquet.rows();
  } else if (options.format == Format::Csv) {
    Input input(fd.value);
    std::vector<Cell> row;
    bool first = true;
    while (csv_row(input, options, row)) {
      if (first) {
        for (size_t i = 0; i < row.size(); ++i) result.columns.push_back({options.header ? row[i].text : "column" + std::to_string(i), Type::Null, false});
        unique_names(result.columns);
        first = false;
        if (options.header) continue;
      }
      require(row.size() == result.columns.size(), "file_sql: CSV column count mismatch at line " + std::to_string(input.line_));
      for (size_t i = 0; i < row.size(); ++i) {
        result.columns[i].type = widen(result.columns[i].type, row[i].type);
        result.columns[i].nullable |= row[i].is_null;
        if (row[i].type == Type::Datetime && row[i].text.size() > 19)
          result.columns[i].scale = std::max<uint8_t>(result.columns[i].scale, row[i].text.size() - 20);
      }
      ++result.rows;
    }
  } else {
    Input input(fd.value);
    ObjectRow row;
    std::unordered_map<std::string, size_t> indices;
    while (json_row(input, row)) {
      std::vector<bool> present(result.columns.size(), false);
      for (const auto &member : row) {
        std::string key = lower(member.first);
        auto found = indices.find(key);
        if (found == indices.end()) {
          require(result.columns.size() < MAX_COLUMNS, "file_sql: JSON union schema exceeds 1024 fields");
          size_t index = result.columns.size();
          indices[key] = index;
          result.columns.push_back({member.first, Type::Null, result.rows != 0});
          present.push_back(false); found = indices.find(key);
        }
        auto &column = result.columns[found->second];
        present[found->second] = true;
        column.type = widen(column.type, member.second.type); column.nullable |= member.second.is_null;
        if (member.second.type == Type::Datetime && member.second.text.size() > 19)
          column.scale = std::max<uint8_t>(column.scale, member.second.text.size() - 20);
      }
      for (size_t i = 0; i < present.size(); ++i) result.columns[i].nullable |= !present[i];
      ++result.rows;
    }
    unique_names(result.columns);
  }
  require(!result.columns.empty(), "file_sql: cannot infer columns from an empty file");
  for (auto &column : result.columns) if (column.type == Type::Null) column.nullable = true;
  result.content_hash = hash_file(fd.value);
  watch.check();
  require(stamp_fd(fd.value) == result.stamp, "file_sql: input changed during schema inference");
  return result;
}

class Scan::Impl
{
public:
  Impl(const Definition &d, const std::string &root, const std::vector<size_t> &projection)
    : definition_(d), root_(root), projection_(projection) { reset(); }
  void check()
  {
    watch_.check();
    require(stamp_fd(fd_.value) == definition_.stamp, "file_sql: input changed since schema inference: " + definition_.path);
    std::string path;
    Fd current(secure_open(definition_.source_path.empty() ? definition_.path : definition_.source_path, root_, path));
    require(stamp_fd(current.value) == definition_.stamp, "file_sql: input path replaced during scan");
  }
  void reset()
  {
    position_ = 0;
    if (definition_.options.format == Format::Glob) { paths_ = discover(definition_.path, root_); return; }
    std::string path;
    if (fd_.value >= 0) ::close(fd_.value);
    fd_.value = -1;
    fd_.value = secure_open(definition_.source_path.empty() ? definition_.path : definition_.source_path, root_, path);
    watch_.reset(path);
    require(hash_file(fd_.value) == definition_.content_hash, "file_sql: input contents changed since schema inference");
    check();
    input_.reset(new Input(fd_.value));
    if (definition_.options.format == Format::Parquet) parquet_.reset(new ParquetScan(fd_.value, definition_.stamp.size, projection_));
    else if (definition_.options.format == Format::Csv && definition_.options.header) { std::vector<Cell> header; csv_row(*input_, definition_.options, header); }
    for (size_t i : projection_) require(i < definition_.columns.size(), "file_sql: invalid projection");
  }
  bool next(std::vector<Cell> &out)
  {
    if (definition_.options.format == Format::Glob) {
      if (position_ == paths_.size()) return false;
      out.clear();
      if (!projection_.empty()) out.push_back({false, paths_[position_], Type::Varchar});
      ++position_; return true;
    }
    check();
    bool available;
    if (parquet_) { bool result = parquet_->next(out); check(); return result; }
    std::vector<Cell> row;
    if (definition_.options.format == Format::Csv) {
      available = csv_row(*input_, definition_.options, row);
      if (available) require(row.size() == definition_.columns.size(), "file_sql: CSV field count changed");
    } else {
      ObjectRow members;
      available = json_row(*input_, members);
      row.resize(definition_.columns.size());
      for (auto &member : members) {
        auto found = std::find_if(definition_.columns.begin(), definition_.columns.end(), [&](const Column &c) { return lower(c.name) == lower(member.first); });
        require(found != definition_.columns.end(), "file_sql: JSON schema changed");
        row[found - definition_.columns.begin()] = std::move(member.second);
      }
    }
    out.clear();
    if (available) {
      for (size_t index : projection_) out.push_back(std::move(row[index]));
      ++position_;
    } else require(position_ == definition_.rows, "file_sql: row count changed");
    check();
    return available;
  }
  Definition definition_;
  std::string root_;
  std::vector<size_t> projection_;
  Fd fd_;
  ChangeWatch watch_;
  std::unique_ptr<Input> input_;
  std::unique_ptr<ParquetScan> parquet_;
  std::vector<std::string> paths_;
  uint64_t position_ = 0;
};

Scan::Scan(const Definition &d, const std::string &root, const std::vector<size_t> &projection)
  : impl_(new Impl(d, root, projection)) {}
Scan::~Scan() = default;
bool Scan::next(std::vector<Cell> &row) { return impl_->next(row); }
void Scan::rescan() { impl_->reset(); }

class CsvWriter::Impl
{
public:
  Impl(const Definition &definition, const std::string &root) : definition_(definition), root_(root)
  {
#ifndef _WIN32
    require(!definition.path.empty() && definition.path.find('\0') == std::string::npos, "file_sql: invalid COPY destination");
    size_t slash = definition.path.rfind('/');
    parent_path_ = slash == std::string::npos ? "." : definition.path.substr(0, slash + 1);
    leaf_ = definition.path.substr(slash == std::string::npos ? 0 : slash + 1);
    require(!leaf_.empty() && leaf_ != "." && leaf_ != "..", "file_sql: invalid COPY filename");
    std::string resolved;
    parent_.value = secure_open(parent_path_, root, resolved, true);
    struct stat st;
    require(::fstatat(parent_.value, leaf_.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0 && errno == ENOENT,
            "file_sql: COPY destination already exists or cannot be accessed");
    require(definition.options.delimiter && definition.options.delimiter != '\r' && definition.options.delimiter != '\n'
            && definition.options.delimiter != definition.options.quote, "file_sql: invalid COPY delimiter");
    require(definition.options.null_string.find_first_of(std::string(1, definition.options.delimiter) + "\r\n"
            + (definition.options.quote ? std::string(1, definition.options.quote) : "")) == std::string::npos,
            "file_sql: NULL text cannot contain CSV control characters");
    static std::atomic<uint64_t> sequence{0};
    for (int attempt = 0; attempt < 128; ++attempt) {
      temporary_ = ".seekdb-copy-" + std::to_string(::getpid()) + "-" + std::to_string(sequence.fetch_add(1));
      int fd = ::openat(parent_.value, temporary_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
      if (fd >= 0) { file_.value = fd; break; }
      require(errno == EEXIST, "file_sql: cannot create COPY temporary file");
    }
    require(file_.value >= 0, "file_sql: COPY temporary-file collision limit exceeded");
#else
    throw Error("file_sql: COPY is not supported on this platform");
#endif
  }
  ~Impl()
  {
#ifndef _WIN32
    if (parent_.value >= 0 && !temporary_.empty()) ::unlinkat(parent_.value, temporary_.c_str(), 0);
#endif
  }
  void write(const std::vector<Cell> &row)
  {
    require(!committed_ && row.size() == definition_.columns.size(), "file_sql: invalid COPY row");
    const Options &options = definition_.options;
    std::string line;
    for (size_t i = 0; i < row.size(); ++i) {
      if (i) line.push_back(options.delimiter);
      if (row[i].is_null) line += options.null_string;
      else {
        const std::string &text = row[i].text;
        bool quote = text.empty() || text == options.null_string || text.find(options.delimiter) != std::string::npos
            || text.find_first_of("\r\n") != std::string::npos || (options.quote && text.find(options.quote) != std::string::npos);
        require(!quote || options.quote, "file_sql: COPY field needs quoting but quote is disabled");
        if (quote) line.push_back(options.quote);
        for (char c : text) {
          if (quote && (c == options.quote || (options.escape != options.quote && c == options.escape))) {
            require(options.escape, "file_sql: COPY field needs escaping but escape is disabled");
            line.push_back(options.escape);
          }
          line.push_back(c);
          require(line.size() <= MAX_ROW, "file_sql: COPY row exceeds 4 MiB");
        }
        if (quote) line.push_back(options.quote);
      }
      require(line.size() <= MAX_ROW, "file_sql: COPY row exceeds 4 MiB");
    }
    line.push_back('\n');
    size_t written = 0;
    while (written < line.size()) {
#ifndef _WIN32
      ssize_t size = ::write(file_.value, line.data() + written, line.size() - written);
      if (size < 0 && errno == EINTR) continue;
      require(size > 0, "file_sql: COPY write failed"); written += size;
#else
      throw Error("file_sql: COPY is not supported on this platform");
#endif
    }
  }
  void commit()
  {
#ifndef _WIN32
    require(!committed_ && ::fsync(file_.value) == 0, "file_sql: COPY flush failed");
    std::string resolved;
    Fd current(secure_open(parent_path_, root_, resolved, true));
    struct stat original, now;
    require(::fstat(current.value, &now) == 0 && ::fstat(parent_.value, &original) == 0
            && original.st_dev == now.st_dev && original.st_ino == now.st_ino, "file_sql: COPY directory changed");
    require(::linkat(parent_.value, temporary_.c_str(), parent_.value, leaf_.c_str(), 0) == 0,
            "file_sql: COPY destination exists or cannot be published");
    ::unlinkat(parent_.value, temporary_.c_str(), 0);
    temporary_.clear(); committed_ = true;
#endif
  }
  Definition definition_;
  std::string root_, parent_path_, leaf_, temporary_;
  Fd parent_, file_;
  bool committed_ = false;
};

CsvWriter::CsvWriter(const Definition &definition, const std::string &root) : impl_(new Impl(definition, root))
{
  if (definition.options.header) {
    std::vector<Cell> header;
    for (const auto &column : definition.columns) header.push_back({false, column.name, Type::Varchar});
    impl_->write(header);
  }
}
CsvWriter::~CsvWriter() = default;
void CsvWriter::write(const std::vector<Cell> &row) { impl_->write(row); }
void CsvWriter::commit() { impl_->commit(); }

}}} // namespace oceanbase::sql::file_sql
