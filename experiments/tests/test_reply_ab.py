"""Local tests: no network, no real model output. Run via unittest discover."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
# Original v10 is installed in user's lab04 app; local package test uses environment path.
import run_evaluation as v10
import compare_replies as ab


class ReplyABTests(unittest.TestCase):
    def test_surface_warnings_are_flags_only(self):
        self.assertIn("suspected_non_uk_words", ab.surface_warnings("Конечно, я подтверждаю.", "uk"))
        self.assertIn("potential_unverified_commitment", ab.surface_warnings("Так, надішлю зараз", "uk"))
        self.assertEqual(ab.surface_warnings(None, "uk"), [])

    def test_cached_source_enforces_matching_config(self):
        class Opt:
            temperature, seed, num_ctx, max_tokens, model = 0, 1, 4096, 220, "qwen3.5:2b-q4_K_M"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            doc = {
                "architecture": "two_stage_classifier_then_optional_reply",
                "model": "qwen3.5:2b-q4_K_M",
                "settings": {"temperature": 0, "seed": 1, "num_ctx": 4096, "max_tokens_per_stage": 220},
                "cases": [{"id":"X01","sender":"a","message":"Привіт","expected_level":"low",
                            "classifier": {"schema_valid": True,"parsed":{"urgency_level":"low","reason":"Вітання"}}}]
            }
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            args = Opt()
            args.source_results = path
            cases, kind, _, _ = ab.read_cases(args)
            self.assertEqual(kind, "cached")
            self.assertEqual(ab.cached_classifier(cases[0])["parsed"]["urgency_level"], "low")
            args.num_ctx = 8192
            with self.assertRaises(ValueError):
                ab.read_cases(args)

    def test_paired_calls_share_same_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                {"id": "U01", "sender": "друг", "message": "Привіт!", "expected_level": "low", "expected_language": "uk"},
                {"id": "U02", "sender": "колега", "message": "До вечора зроби файл", "expected_level": "medium", "expected_language": "uk"},
            ]
            case_file = root / "cases.json"
            case_file.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            prompt_a = root / "a.txt"
            prompt_b = root / "b.txt"
            prompt_c = root / "c.txt"
            prompt_a.write_text("PROMPT_A", encoding="utf-8")
            prompt_b.write_text("PROMPT_B", encoding="utf-8")
            prompt_c.write_text("CLASSIFIER", encoding="utf-8")
            sent = []

            def fake_request(args, *, prompt, case_data, schema, max_tokens):
                sent.append((prompt, dict(case_data)))
                metric = {"elapsed_s": .5, "prompt_tokens": 50, "output_tokens": 9,
                          "total_tokens": 59, "done_reason": "stop"}
                if schema is v10.CLASS_SCHEMA:
                    lvl = "medium" if case_data["message"].startswith("До") else "low"
                    return json.dumps({"urgency_level": lvl, "reason": "Зазначено дедлайн"}), metric
                if prompt == "PROMPT_A":
                    return '{"suggested_reply":"Звісно, надішлю."}', metric
                return '{"suggested_reply":"Бачу, що це потрібно до вечора."}', metric

            argv = ["--label", "mock_ua", "--cases-file", str(case_file),
                    "--prompt-a", str(prompt_a), "--prompt-b", str(prompt_b),
                    "--classifier-prompt", str(prompt_c), "--no-warmup"]
            with patch.object(ab, "ROOT", root), patch.object(v10, "request_stage", fake_request):
                with contextlib.redirect_stdout(io.StringIO()):
                    ab.main(argv)
            data = json.loads((root / "experiments" / "results" / "reply_ab_mock_ua.json").read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["category_correct"], 2)
            self.assertEqual(data["summary"]["A"]["attempted"], 2)
            self.assertEqual(data["summary"]["B"]["attempted"], 2)
            self.assertEqual(data["cases"][0]["run_order"], "A then B")
            self.assertEqual(data["cases"][1]["run_order"], "B then A")
            for i, case in enumerate(cases):
                inputs = [v for p, v in sent if p != "CLASSIFIER" and v["message"] == case["message"]]
                self.assertEqual(len(inputs), 2)
                self.assertEqual(inputs[0], inputs[1])
                self.assertNotIn("expected_level", inputs[0])
                self.assertNotIn("manual_goal", inputs[0])
            self.assertTrue((root / "experiments" / "results" / "reply_ab_mock_ua.csv").exists())


if __name__ == "__main__":
    unittest.main()
