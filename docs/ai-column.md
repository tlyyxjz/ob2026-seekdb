# Asynchronous AI columns

Register an AI model and endpoint with `DBMS_AI_SERVICE` before declaring an
AI column. MySQL mode supports `AI_COMPLETE` results in JSON, integer, numeric,
string, DATE and DATETIME columns, and `AI_EMBED` results in VECTOR columns.
Only `sync_mode='async'` is supported. AI columns cannot depend on other AI
columns or be directly assigned by business DML.

```sql
CREATE TABLE documents (
  id BIGINT PRIMARY KEY,
  title VARCHAR(100),
  body TEXT,
  tags JSON AI COLUMN (AI_COMPLETE('text_model', body))
    WITH (sync_mode='async', max_retries=2),
  embedding VECTOR(4) AI COLUMN (AI_EMBED('embed_model', body, 4))
    WITH (sync_mode='async', max_retries=2)
);
INSERT INTO documents(id, title, body) VALUES (1, 'Example', 'Document text');
SHOW AI COLUMNS FROM documents;
ALTER TABLE documents RETRY AI COLUMN tags WHERE status='FAILED';
SET SESSION ai_column_strict_read=ON;
SELECT tags FROM documents WHERE id=1;
```

An explicit embedding dimension is checked against the column and passed through
to the provider. Completion calls receive a system instruction for the declared
result type; the business input expression is captured at the row change.

Each AI column has three hidden physical columns for its state, request token
and definition version. `oceanbase.__all_ai_column` exposes its definition and
hidden-column IDs. States are PENDING, READY and FAILED; READY is independent of
whether the stored result is SQL NULL. Heap tables also have a hidden, indexed
row UUID assigned by the native BEFORE trigger, so task ownership does not depend
on a physical heap key that vector maintenance can replace. A NULL input follows scalar function NULL
semantics without an external model request.

Native row triggers invalidate dependent results and insert durable tasks into
`oceanbase.__all_ai_column_task` in the business transaction. An unrelated column
update leaves results and tasks unchanged. A primary-key update invalidates all
AI columns. Deletes remove outstanding tasks; rolled-back writes leave neither
tasks nor model requests.

Four native background workers claim committed request groups with database
locks and leases. Model calls run after the claim transaction commits. Retries
are finite and use bounded backoff. All columns invalidated by one row change
share a request token. Successful typed results are retained in the queue until
every current group member reaches READY or final failure. One native UPDATE
publishes results and states, with queue deletion in the same transaction.
Request token, definition version and task ownership checks prevent stale writes
and resurrection of deleted rows. An interrupted in-flight inference can be
repeated after restart; already stored terminal results can be published without
another inference.

Vector invalidation and publication use normal transactional DML and native
index maintenance. BEFORE triggers that can change a vector also refresh its
native VID, including when the original SET list only names an input column.

Strict reads check only referenced AI columns, across every row in the same
statement snapshot. Their internal read shares the caller transaction, including
prior uncommitted invalidations. Ordinary columns and other READY AI columns
remain readable. A not-ready referenced column raises MySQL error 11120 before
the main operator runs, including UPDATE and DELETE predicates. Bare `status`
in RETRY is a state pseudo column; backticked `status` is a business column.

Public integration cases are under `tools/deploy/mysql_test/test_suite/ai_column`:
`ai_column_ddl`, `ai_column_t1_dml`, `ai_column_worker`,
`ai_column_retry_strict`, and `ai_column_vector_index`. Run them with the normal
mysqltest runner from `tools/deploy` so the included Mock service paths resolve.
The additional `ai_column_heap` regression covers heap identity, mixed terminal
types, strict INSERT, vector invalidation, rollback and regeneration.
Parser, resolver and schema regression tests are in the corresponding
`unittest/sql` and `unittest/share/schema` directories.
