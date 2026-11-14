"""Quick offline contract tests. No network or real Ollama inference."""
from __future__ import annotations
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import run_evaluation as runner


class TestValidation(unittest.TestCase):
    def test_classifier(self):
        self.assertEqual(runner.validate_classifier({"urgency_level": "high", "reason": "  deadline "}),
                         {"urgency_level": "high", "reason": "deadline"})
        with self.assertRaises(ValueError):
            runner.validate_classifier({"urgency_level": "high", "reason": "ok", "suggested_reply": None})
        with self.assertRaises(ValueError):
            runner.validate_classifier({"urgency_level": "immediate", "reason": "ok"})

    def test_reply_null_type(self):
        self.assertEqual(runner.validate_reply({"suggested_reply": None}), (None, None))
        self.assertEqual(runner.validate_reply({"suggested_reply": "null"}),
                         ("null", "string_null_instead_of_json_null"))
        self.assertEqual(runner.validate_reply({"suggested_reply": "  "}),
                         ("  ", "empty_string_instead_of_null"))
        with self.assertRaises(ValueError):
            runner.validate_reply({"suggested_reply": 3})

    def test_runner_mock_two_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cases = [
                {"id": "X01", "sender": "demo", "message": "Hi", "expected_level": "low", "expected_language": "en"},
                {"id": "X02", "sender": "demo", "message": "Help right now", "expected_level": "critical", "expected_language": "en"},
            ]
            file = directory / "cases.json"
            file.write_text(json.dumps(cases), encoding="utf-8")
            call_count = [0]

            def pretend_request(args, *, prompt, case_data, schema, max_tokens):
                call_count[0] += 1
                metrics = {"elapsed_s": 0.5, "prompt_tokens": 60, "output_tokens": 15,
                           "total_tokens": 75, "done_reason": "stop"}
                if schema is runner.CLASS_SCHEMA:
                    level = "critical" if case_data["message"].startswith("Help") else "low"
                    return json.dumps({"urgency_level": level, "reason": "test"}), metrics
                # Deliberately invalid *semantic* null in second response
                return json.dumps({"suggested_reply": "null"}), metrics

            argv = ["program", "--label", "mock", "--cases-file", str(file), "--no-warmup"]
            with patch.object(runner, "ROOT", directory), \
                 patch.object(runner, "request_stage", side_effect=pretend_request), \
                 patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(call_count[0], 4)
            report = json.loads((directory / "experiments" / "results" / "evaluation_mock.json").read_text())
            self.assertEqual(report["summary"]["category_correct"], 2)
            self.assertEqual(report["summary"]["reply_schema_valid"], 2)
            self.assertEqual(report["summary"]["reply_semantic_valid"], 0)
            self.assertEqual(report["summary"]["reply_string_null_errors"], 2)
            self.assertEqual(report["summary"]["mean_total_elapsed_s"], 1.0)
            self.assertEqual(report["summary"]["mean_total_prompt_tokens"], 120)
            self.assertTrue((directory / "experiments" / "results" / "evaluation_mock.csv").exists())


if __name__ == "__main__":
    unittest.main()
