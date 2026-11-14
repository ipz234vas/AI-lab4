"""SmartNotify: paired A/B for two reply prompts with SAME classifier output.

Requires experiments/run_evaluation.py. Standard library only.
Does not send replies or edit the existing v10 experiment artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import run_evaluation as v10

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_A = HERE / "prompts" / "reply_candidate.txt"
DEFAULT_B = ROOT / "app" / "prompts" / "reply.txt"
DEFAULT_CASES = HERE / "cases" / "quality_10.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_cases(args):
    if args.source_results:
        doc = json.loads(args.source_results.read_text(encoding="utf-8-sig"))
        if doc.get("architecture") != "two_stage_classifier_then_optional_reply":
            raise ValueError("--source-results має бути JSON звітом v10 split")
        settings = doc.get("settings", {})
        for key, value in (("temperature", args.temperature), ("seed", args.seed),
                           ("num_ctx", args.num_ctx), ("max_tokens_per_stage", args.max_tokens)):
            if settings.get(key) != value:
                raise ValueError(f"Параметр {key} у cached звіті ({settings.get(key)}) відрізняється від поточного ({value})")
        if doc.get("model") != args.model:
            raise ValueError("Cached JSON використовує іншу модель; не можна тихо порівнювати")
        cases = doc.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("Не знайдено cases у cached JSON")
        return cases, "cached", doc.get("cases_sha256"), doc.get("classifier_prompt_sha256")
    if not args.cases_file.exists():
        raise ValueError(f"Не знайдено кейси: {args.cases_file}")
    content = args.cases_file.read_bytes()
    cases = json.loads(content.decode("utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("--cases-file має містити непорожній масив JSON")
    return cases, "new", sha(content), None


def validate_case(case):
    if not isinstance(case, dict):
        raise ValueError("Кейс не є об'єктом")
    for key in ("id", "sender", "message", "expected_level"):
        if key not in case or not isinstance(case[key], str) or not case[key].strip():
            raise ValueError(f"Кейс без {key}")
    if case["expected_level"] not in v10.LEVELS:
        raise ValueError("Некоректний expected_level")


def cached_classifier(case):
    raw = case.get("classifier", {})
    if not raw.get("schema_valid"):
        return {"schema_valid": False, "error": "cached_classifier_invalid"}
    parsed = raw.get("parsed")
    if not isinstance(parsed, dict):
        # Old v10 parsed stage classifier is dict, reject anything else.
        raise ValueError(f"Для cached case {case['id']} немає classified parsed dict")
    return {"schema_valid": True, "parsed": v10.validate_classifier(parsed),
            "metrics": raw.get("metrics", {}), "cached": True}


def surface_warnings(value, expected_language):
    """Only observable flags. NOT a grammar / meaning quality score."""
    if value is None:
        return []
    if not isinstance(value, str):
        return ["not_string_or_null"]
    x = value.strip()
    warnings = []
    if not x or x.casefold() in v10.INVALID_NULL_LITERALS:
        warnings.append("empty_or_string_null")
    if len(x) > 320:
        warnings.append("long_draft")
    if "як штучний інтелект" in x.casefold() or "як ші" in x.casefold() or "as an ai" in x.casefold():
        warnings.append("ai_meta_language")
    # Very coarse manual-review flags; not automatic hard failures:
    if expected_language == "uk" and re.search(r"\b(?:конечно|пожалуйста|понятно|отправлю|сегодня|завтра|привет|я готово)\b", x, flags=re.I):
        warnings.append("suspected_non_uk_words")
    if re.search(r"\b(?:надішлю|відправлю|зроблю|підтверджую|погоджую|оплачу|викличу|перевірю|зателефоную|надсилаю)\b", x, flags=re.I):
        warnings.append("potential_unverified_commitment")
    return warnings


def pull_reply(stage):
    if not stage.get("schema_valid"):
        return None, "invalid_schema"
    draft, sem_error = stage.get("parsed", (None, None))
    return draft, sem_error


def stage_view(stage, expected_language):
    draft, error = pull_reply(stage)
    metrics = stage.get("metrics", {})
    return {"schema_valid": stage.get("schema_valid", False),
            "draft": draft,
            "semantic_error": error,
            "surface_warnings": surface_warnings(draft, expected_language),
            "metrics": metrics,
            "error": stage.get("error"),
            "raw_answer": stage.get("raw_answer")}


def mean(data):
    vals = [x for x in data if isinstance(x, (float, int))]
    return round(sum(vals) / len(vals), 3) if vals else None


def aggregate(rows, key):
    stages = [row.get(key, {}) for row in rows if not row.get("skipped")]
    valid = [s for s in stages if s.get("schema_valid")]
    return {
        "attempted": len(stages), "schema_valid": len(valid),
        "nonnull": sum(isinstance(s.get("draft"), str) for s in valid),
        "json_null": sum(s.get("draft") is None for s in valid),
        "null_string_error": sum(s.get("semantic_error") == "string_null_instead_of_json_null" for s in valid),
        "mean_elapsed_s": mean(s.get("metrics", {}).get("elapsed_s") for s in stages),
        "mean_prompt_tokens": mean(s.get("metrics", {}).get("prompt_tokens") for s in stages),
        "mean_output_tokens": mean(s.get("metrics", {}).get("output_tokens") for s in stages),
        "surface_flags": sum(len(s.get("surface_warnings", [])) for s in valid),
    }


def report_summary(rows):
    return {
        "total": len(rows),
        "valid_classifier": sum(bool(r.get("classifier_valid")) for r in rows),
        "category_correct": sum(bool(r.get("level_correct")) for r in rows),
        "A": aggregate(rows, "A"), "B": aggregate(rows, "B"),
        "NOTE": "Naturalness, roles, grounding, useful draft and emergency safety REQUIRE manual review; no LLM-as-judge score is claimed.",
    }


def write_csv(path, rows):
    cols = ["id", "expected_language", "message", "expected_level", "predicted_level",
            "classifier_valid", "level_correct", "reply_policy", "manual_goal", "run_order",
            "A_reply", "B_reply", "A_json_valid", "B_json_valid", "A_semantic_error", "B_semantic_error",
            "A_surface_warnings", "B_surface_warnings", "A_elapsed_s", "B_elapsed_s",
            "A_prompt_tokens", "B_prompt_tokens", "A_output_tokens", "B_output_tokens",
            "A_ukrainian_ok", "B_ukrainian_ok", "A_role_ok", "B_role_ok", "A_facts_ok", "B_facts_ok",
            "A_no_false_promise", "B_no_false_promise", "A_natural", "B_natural", "A_useful", "B_useful",
            "A_safe", "B_safe", "winner_A_B_tie_neither", "review_notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            row = {name: "" for name in cols}
            for name in ("id", "message", "expected_level", "expected_language", "predicted_level", "classifier_valid", "level_correct", "reply_policy", "manual_goal", "run_order"):
                row[name] = r.get(name, "")
            for stage in ("A", "B"):
                s = r.get(stage, {})
                row[f"{stage}_reply"] = s.get("draft")
                row[f"{stage}_json_valid"] = s.get("schema_valid")
                row[f"{stage}_semantic_error"] = s.get("semantic_error")
                row[f"{stage}_surface_warnings"] = "; ".join(s.get("surface_warnings", []))
                m = s.get("metrics", {})
                row[f"{stage}_elapsed_s"] = m.get("elapsed_s")
                row[f"{stage}_prompt_tokens"] = m.get("prompt_tokens")
                row[f"{stage}_output_tokens"] = m.get("output_tokens")
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description="A/B comparison of two reply prompts")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-results", type=Path, help="Cache classifier from existing v10 JSON, run ONLY two generators")
    source.add_argument("--cases-file", type=Path, default=None, help="New cases: classify ONCE, run BOTH generators")
    parser.add_argument("--prompt-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--prompt-b", type=Path, default=DEFAULT_B)
    parser.add_argument("--classifier-prompt", type=Path, default=ROOT / "app" / "prompts" / "classifier.txt")
    parser.add_argument("--model", default="qwen3.5:2b-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--label", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)
    if args.cases_file is None and not args.source_results:
        args.cases_file = DEFAULT_CASES
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("--label: латиниця/цифри/underscore/hyphen")
    if not (0 <= args.temperature <= 2 and args.num_ctx >= 512 and args.max_tokens >= 48):
        parser.error("Невірні num_ctx / max_tokens / temperature")
    try:
        pA = args.prompt_a.read_text(encoding="utf-8-sig")
        pB = args.prompt_b.read_text(encoding="utf-8-sig")
        pC = args.classifier_prompt.read_text(encoding="utf-8-sig") if not args.source_results else ""
        cases, source_kind, cases_hash, old_class_hash = read_cases(args)
        if not all((pA.strip(), pB.strip())) or (source_kind == "new" and not pC.strip()):
            raise ValueError("Промпт порожній")
        for c in cases:
            validate_case(c)
        if len({c["id"] for c in cases}) != len(cases):
            raise ValueError("Унікальність ID порушено")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
        parser.error(str(e))

    out = ROOT / "experiments" / "results" / f"reply_ab_{args.label}.json"
    csv_path = out.with_suffix(".csv")
    if args.check_only:
        print(f"OK: {len(cases)} cases; source={source_kind}; same classifier for A/B; model={args.model}")
        print(f"cases sha256={cases_hash}; prompt A={sha(pA.encode())}; prompt B={sha(pB.encode())}")
        return
    if out.exists() or csv_path.exists():
        parser.error(f"Результат із label={args.label} вже існує")
    out.parent.mkdir(exist_ok=True, parents=True)

    report = {
        "experiment": "v10.1_same_classifier_paired_reply_ab",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source_kind": source_kind,
        "source_path": str(args.source_results if args.source_results else args.cases_file),
        "cases_sha256": cases_hash,
        "classifier_prompt_sha256": old_class_hash if source_kind == "cached" else sha(pC.encode()),
        "prompt_A": {"path": str(args.prompt_a), "sha256": sha(pA.encode())},
        "prompt_B": {"path": str(args.prompt_b), "sha256": sha(pB.encode())},
        "settings": {"temperature": args.temperature, "seed": args.seed, "num_ctx": args.num_ctx,
                     "max_tokens_per_stage": args.max_tokens, "think": False, "format": "json_schema"},
        "caveat": "The source classifier is fixed for BOTH reply prompts; no holdout is run. Quality requires manual Ukrainian review.",
        "warmup": {}, "cases": [], "summary": {},
    }

    if not args.no_warmup:
        # Warm both generators using the same warmup, not counted in the measurements.
        for key, prompt in (("A", pA), ("B", pB)):
            try:
                _, metrics = v10.request_stage(args, prompt=prompt,
                            case_data={"sender": "friend", "message": "Привіт!", "urgency_level": "low"},
                            schema=v10.REPLY_SCHEMA, max_tokens=args.max_tokens)
                report["warmup"][key] = metrics
            except Exception as e:
                report["warmup"][key] = {"error": f"{type(e).__name__}: {e}"}
        if source_kind == "new":
            try:
                _, metrics = v10.request_stage(args, prompt=pC,
                            case_data={"sender": "friend", "message": "Привіт!"},
                            schema=v10.CLASS_SCHEMA, max_tokens=args.max_tokens)
                report["warmup"]["classifier"] = metrics
            except Exception as e:
                report["warmup"]["classifier"] = {"error": f"{type(e).__name__}: {e}"}

    for i, case in enumerate(cases):
        if source_kind == "cached":
            c = cached_classifier(case)
        else:
            c = v10.run_stage(args, pC, {"sender": case["sender"], "message": case["message"]},
                              v10.CLASS_SCHEMA, args.max_tokens, v10.validate_classifier)
        level = c.get("parsed", {}).get("urgency_level") if c.get("schema_valid") else None
        rec = {"id": case["id"], "sender": case["sender"], "message": case["message"],
               "expected_level": case["expected_level"],
               "expected_language": case.get("expected_language"),
               "reply_policy": case.get("reply_policy"),
               "manual_goal": case.get("manual_goal"),
               "predicted_level": level, "classifier_valid": bool(c.get("schema_valid")),
               "level_correct": level == case["expected_level"],
               "classifier": c,
               "run_order": "A then B" if i % 2 == 0 else "B then A"}
        if c.get("schema_valid"):
            payload = {"sender": case["sender"], "message": case["message"], "urgency_level": level}
            for key in (("A", "B") if i % 2 == 0 else ("B", "A")):
                prompt = pA if key == "A" else pB
                stage = v10.run_stage(args, prompt, payload, v10.REPLY_SCHEMA, args.max_tokens, v10.validate_reply)
                rec[key] = stage_view(stage, case.get("expected_language"))
        else:
            rec.update(skipped=True, A={}, B={})
        report["cases"].append(rec)
        report["summary"] = report_summary(report["cases"])
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{case['id']} [{level or 'ERROR':8s}] A: {str(rec['A'].get('draft'))[:50]!r} | B: {str(rec['B'].get('draft'))[:50]!r}", flush=True)

    write_csv(csv_path, report["cases"])
    print("\nClassifier:", report["summary"]["category_correct"], "/", len(cases))
    for tag in ("A", "B"):
        s = report["summary"][tag]
        print(f"{tag} ({'v10 baseline' if tag=='A' else 'v10.1 Ukrainian'}): "
              f"{s['schema_valid']}/{s['attempted']} JSON; {s['json_null']} null; "
              f"{s['mean_elapsed_s']}s; {s['mean_prompt_tokens']} prompt tokens")
    print("JSON:", out, "\nCSV (review manually):", csv_path)


if __name__ == "__main__":
    main()
