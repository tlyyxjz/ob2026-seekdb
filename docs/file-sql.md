# Local file SQL

File SQL is disabled by default. Enable it for a session, and configure
`secure_file_priv` to an allowed local directory using the normal server
administration procedure. Empty and `NULL` settings disable file access,
matching seekdb's existing path restriction contract. The current global
restriction is checked again when execution starts, including cached plans.

```sql
SET SESSION enable_file_sql = ON;
SELECT id, SUM(amount)
FROM read_csv('/allowed/orders.csv', header = true, sample_size = -1) AS orders
GROUP BY id;

SELECT c.id, j.name
FROM read_csv('/allowed/a.csv') AS c
LEFT JOIN read_ndjson('/allowed/b.jsonl') AS j ON c.id = j.id;

DESCRIBE SELECT * FROM read_parquet('/allowed/data.parquet');
SELECT file FROM glob('/allowed/*.csv') ORDER BY file;

COPY (SELECT * FROM read_csv('/allowed/orders.csv') ORDER BY id)
TO '/allowed/export.csv' (FORMAT CSV, HEADER, NULLSTR 'NULL');
```

The file functions are native function-table scans. Expressions, predicates,
joins, sorting, and aggregation use seekdb's SQL execution engine. Input data
is not imported into a table or retained as an in-memory table. CSV and JSONL
schema inference reads the entire input; only `sample_size = -1` is supported.
Each new statement infers its schema afresh and bypasses the plan cache;
subsequent operator rescans stream records. Parquet scans decode selected columns.

CSV options are `header`, `delim` (or `delimiter`), `quote`, `escape`, and
`nullstr`. Separators and quoting characters are single-byte characters.
Quoted empty strings and quoted NULL text remain strings. Newlines inside
CSV quotes belong to the same record; CRLF is normalized to LF. JSONL accepts
flat objects, unions their fields in first-seen order, and fills missing
fields with NULL. Nested objects and arrays are rejected. `read_json` and
`read_ndjson` are equivalent newline-delimited readers.

Inference supports NULL, BOOLEAN, BIGINT, DOUBLE, VARCHAR, DATE, and DATETIME.
Numeric types widen through BOOLEAN, BIGINT, and DOUBLE; DATE and DATETIME
widen to DATETIME. Other conflicts become VARCHAR. Integers outside signed
64-bit range remain VARCHAR. Date validation includes leap years, and
DATETIME preserves up to six fractional digits.

The native Parquet reader supports flat optional/required primitive columns,
multiple row groups, PLAIN and dictionary encodings, boolean RLE, v1/v2 data
pages, and UNCOMPRESSED/SNAPPY compression. Nested/repeated columns, decimal,
and other codecs/encodings fail with an explicit diagnostic.

Reads check canonical paths against `secure_file_priv`, open components
without following symlinks after validation, and reject non-regular files.
Scans validate file identity, timestamps, and an opening content fingerprint;
Linux also monitors writes during inference and scanning. Changed inputs
fail instead of silently reusing stale inferred types. Limits include 1,024
columns, 4 MiB per text record, 10,000 glob results, 16 MiB Parquet footers,
8 MiB Parquet pages, and 64 MiB decoded projected Parquet pages.

COPY supports the CSV output options above, defaults to no header, and never
replaces an existing target. It streams to a private temporary file in the
validated destination directory, then publishes without overwrite after the
query succeeds. Errors clean up the temporary file. Linux and macOS are the
supported file-I/O platforms for this implementation.
