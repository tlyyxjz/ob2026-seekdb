"""Offline SQL regression tests against a running candidate seekdb instance.

Run with Python 3, passing --client=/path/to/obclient and --port=<SQL port>.
The mock HTTP service runs in this process; no model or third-party API is used.
"""

import argparse
import json
import re
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def vector(text, dimension):
    return [float(sum(text.encode("utf-8")))] + [0.25] * (dimension - 1)


class MockHandler(BaseHTTPRequestHandler):
    events = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.events.append((self.path, body))
        status = 200
        if self.path.startswith("/error/"):
            status, result = 400, {"error": "injected failure"}
        elif self.path.endswith(("/embeddings", "/embed")):
            texts = body["input"]
            dimension = body.get("dimensions", 768)
            embeddings = [vector(text, dimension) for text in texts]
            if self.path.startswith("/wrong-dimension/"):
                embeddings = [item + [0] for item in embeddings]
            if self.path.endswith("/embed"):
                result = {"embeddings": embeddings}
            else:
                data = [{"index": i, "embedding": item} for i, item in enumerate(embeddings)]
                if self.path.startswith("/duplicate-index/") and len(data) > 1:
                    data[-1]["index"] = 0
                result = {"data": list(reversed(data))}
        elif self.path.endswith("/rerank"):
            terms = set(re.findall(r"\w+", body["query"].lower()))
            results = [
                {"index": i, "relevance_score": len(terms & set(re.findall(r"\w+", text.lower())))}
                for i, text in enumerate(body["documents"])
            ]
            result = {"results": list(reversed(results))}
        else:
            text = body.get("prompt")
            if text is None:
                text = body["messages"][-1]["content"]
            content = None if self.path.startswith("/empty/") else "mock:" + text
            result = {"response": content} if self.path.endswith("/generate") else {
                "choices": [{"message": {"content": content}}]
            }
        encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class AIFunctionSQLTest(unittest.TestCase):
    @classmethod
    def sql(cls, statement, expect_error=False):
        process = subprocess.run(
            [OPTIONS.client, "-h127.0.0.1", "-P" + str(OPTIONS.port), "-uroot", "-N", "-B",
             "--raw", "--default-character-set=utf8mb4", "-e", statement],
            capture_output=True, text=True, timeout=60,
        )
        if expect_error:
            if process.returncode == 0:
                raise AssertionError("SQL unexpectedly succeeded: " + statement)
        elif process.returncode:
            raise AssertionError(process.stderr + "\nSQL: " + statement)
        return process.stdout.strip()

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.names = []
        cls.sql("create database if not exists ai_function_regression")
        definitions = [
            ("embed", "dense_embedding", "openai", "/embeddings"),
            ("complete", "completion", "openai", "/chat/completions"),
            ("rerank", "rerank", "siliconflow", "/rerank"),
            ("ollama", "completion", "oLlAmA", "/api/generate"),
            ("ollama_embed", "dense_embedding", "OLLAMA", "/api/embed"),
            ("wrong", "dense_embedding", "openai", "/wrong-dimension/embeddings"),
            ("duplicate", "dense_embedding", "openai", "/duplicate-index/embeddings"),
            ("empty", "completion", "openai", "/empty/chat/completions"),
            ("error", "dense_embedding", "openai", "/error/embeddings"),
        ]
        for suffix, model_type, provider, path in definitions:
            name = "ai_http_reg_" + suffix
            cls.names.append(name)
            cls.sql("call DBMS_AI_SERVICE.CREATE_AI_MODEL('%s','%s')" % (
                name, json.dumps({"type": model_type, "model_name": "regression-model"})))
            endpoint = {"ai_model_name": name, "scope": "all", "provider": provider,
                        "url": "http://127.0.0.1:%d%s" % (cls.server.server_port, path),
                        "access_key": "" if provider.lower() == "ollama" else "mock"}
            cls.sql("call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('%s_ep','%s')" % (
                name, json.dumps(endpoint)))

    @classmethod
    def tearDownClass(cls):
        for name in cls.names:
            cls.sql("call DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT('%s_ep')" % name)
            cls.sql("call DBMS_AI_SERVICE.DROP_AI_MODEL('%s')" % name)
        cls.sql("drop database ai_function_regression")
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_prompt_json_repetition_nested_and_long_text(self):
        result = self.sql("select ai_complete('ai_http_reg_complete',"
            "ai_prompt('{0}/{0}/{1}',json_array(1,true),ai_prompt('inner {0}','你好')),null)")
        self.assertEqual("mock:[1, true]/[1, true]/inner 你好", result)
        self.sql("use ai_function_regression; create table prompts(t longtext,a text);"
                 "insert into prompts values('{0}/{0}',repeat('x',10000))")
        result = self.sql("select ai_complete('ai_http_reg_complete',ai_prompt(t,a),"
            "json_object('temperature',0.2,'max_tokens',16)) from ai_function_regression.prompts")
        self.assertEqual("mock:" + "x" * 10000 + "/" + "x" * 10000, result)
        self.sql("select ai_complete('ai_http_reg_complete',ai_prompt('{9}','a'))", expect_error=True)

    def test_scalar_embeddings_dimension_and_invalid_results(self):
        self.assertEqual("768", self.sql("select json_length(ai_embed('ai_http_reg_embed','x'))"))
        self.assertEqual(vector("你好", 4), json.loads(self.sql("select ai_embed('ai_http_reg_embed','你好',4)")))
        for dimension in ("0", "-1", "1.5", "null", "'4'"):
            self.sql("select ai_embed('ai_http_reg_embed','x',%s)" % dimension, expect_error=True)
        self.sql("select ai_embed('ai_http_reg_wrong','x',4)", expect_error=True)
        self.sql("select ai_embed('ai_http_reg_error','x',4)", expect_error=True)

    def test_batch_reversed_responses_null_filter_limit_and_multiple_requests(self):
        self.sql("use ai_function_regression; create table inputs(id int primary key,t text); insert into inputs values " +
                 ",".join("(%d,%s)" % (i, "null" if i % 7 == 0 else "'row%d'" % i) for i in range(75)))
        start = len(MockHandler.events)
        rows = self.sql("select id,ai_embed('ai_http_reg_embed',t,4) from ai_function_regression.inputs "
                        "where id>=3 order by id limit 65").splitlines()
        self.assertEqual(65, len(rows))
        for row in rows:
            identity, encoded = row.split("\t", 1)
            i = int(identity)
            self.assertEqual("NULL", encoded) if i % 7 == 0 else self.assertEqual(vector("row%d" % i, 4), json.loads(encoded))
        batches = [body["input"] for path, body in MockHandler.events[start:] if path == "/embeddings"]
        self.assertTrue(any(len(batch) > 1 for batch in batches), "cross-row requests were not merged")
        self.assertTrue(all(len(batch) <= 32 for batch in batches))
        self.assertGreater(len(batches), 1)
        self.sql("select ai_embed('ai_http_reg_duplicate',t,4) from ai_function_regression.inputs "
                 "where t is not null limit 3", expect_error=True)

    def test_rerank_global_order_and_original_objects(self):
        documents = ["irrelevant"] * 21 + ["needle", "needle"]
        result = json.loads(self.sql("select ai_rerank('ai_http_reg_rerank','needle','%s',null)" % json.dumps(documents)))
        self.assertEqual([21, 22] + list(range(21)), [row["index"] for row in result])
        result = json.loads(self.sql("select ai_rerank('ai_http_reg_rerank','needle',"
            "'[{} , {}]','text')".format(json.dumps({"id": "a", "text": "other"}), json.dumps({"id": "b", "text": "needle"}))))
        self.assertEqual(["b", "a"], [row["id"] for row in result])

    def test_ollama_and_completion_failure(self):
        self.sql("call DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT('ai_http_reg_ollama_ep',"
                 "'{\"provider\":\"Ollama\"}')")
        self.assertEqual("mock:hello", self.sql("select ai_complete('ai_http_reg_ollama','hello')"))
        body = next(body for path, body in reversed(MockHandler.events) if path == "/api/generate")
        self.assertIs(False, body["stream"])
        self.assertEqual(4, len(json.loads(self.sql("select ai_embed('ai_http_reg_ollama_embed','hello',4)"))))
        self.sql("select ai_complete('ai_http_reg_empty','hello')", expect_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--port", type=int, default=2881)
    OPTIONS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__] + remaining, verbosity=2)
