"""SmartNotify research runner: two independent Ollama /api/chat stages, standard-library only.

Stage 1 classifies urgency and explains; Stage 2 optionally drafts a reply.
No message is sent to the original sender. Critical alerts must be handled by
application UI independent of the optional draft.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEVELS = ("low", "medium", "high", "critical")
DEFAULT_CLASSIFIER = ROOT / "app" / "prompts" / "classifier.txt"
DEFAULT_REPLY = ROOT / "app" / "prompts" / "reply.txt"
CLASS_SCHEMA = {
    "type": "object",
    "properties": {
        "urgency_level": {"type": "string", "enum": list(LEVELS)},
        "reason": {"type": "string"},
    },
    "required": ["urgency_level", "reason"],
    "additionalProperties": False,
}
REPLY_SCHEMA = {
    "type": "object",
    "properties": {"suggested_reply": {"type": ["string", "null"]}},
    "required": ["suggested_reply"],
    "additionalProperties": False,
}
INVALID_NULL_LITERALS = {"null", "none", "n/a", "undefined"}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seconds(nanoseconds):
    return round(nanoseconds / 1e9, 4) if isinstance(nanoseconds, (int, float)) else None


def validate_classifier(data):
    if not isinstance(data, dict) or set(data) != set(CLASS_SCHEMA["required"]):
        raise ValueError("Класифікатор: очікуються рівно urgency_level і reason")
    if data["urgency_level"] not in LEVELS or not isinstance(data["urgency_level"], str):
        raise ValueError("Класифікатор: некоректна urgency_level")
    if not isinstance(data["reason"], str) or not data["reason"].strip():
        raise ValueError("Класифікатор: reason має бути непорожнім рядком")
    return {"urgency_level": data["urgency_level"], "reason": data["reason"].strip()}


def validate_reply(data):
    if not isinstance(data, dict) or set(data) != {"suggested_reply"}:
        raise ValueError("Генератор: очікується тільки suggested_reply")
    draft = data["suggested_reply"]
    if draft is not None and not isinstance(draft, str):
        raise ValueError("Генератор: suggested_reply має бути string або null")
    # Do not silently replace the model's literal "null" with true JSON null.
    semantic_error = None
    if isinstance(draft, str):
        if not draft.strip():
            semantic_error = "empty_string_instead_of_null"
        elif draft.strip().casefold() in INVALID_NULL_LITERALS:
            semantic_error = "string_null_instead_of_json_null"
    return draft, semantic_error


def request_stage(args, *, prompt, case_data, schema, max_tokens):
    url = args.base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(case_data, ensure_ascii=False)},
        ],
        "stream": False,
        "think": False,
        "format": schema,
        "keep_alive": "10m",
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_ctx": args.num_ctx,
            "num_predict": max_tokens,
        },
    }
    req = Request(url,
                  data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urlopen(req, timeout=args.timeout) as resp:
        result = json.load(resp)
    elapsed = round(time.perf_counter() - t0, 3)
    output = result.get("message", {}).get("content", "")
    prompt_tokens = result.get("prompt_eval_count")
    output_tokens = result.get("eval_count")
    decode = seconds(result.get("eval_duration"))
    metrics = {
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens
            if isinstance(prompt_tokens, int) and isinstance(output_tokens, int) else None,
        "load_s": seconds(result.get("load_duration")),
        "prefill_s": seconds(result.get("prompt_eval_duration")),
        "decode_s": decode,
        "generation_tokens_per_s": round(output_tokens / decode, 2)
            if isinstance(output_tokens, int) and decode and decode > 0 else None,
        "done_reason": result.get("done_reason"),
    }
    return output, metrics


def describe_mean(values):
    numbers = [x for x in values if isinstance(x, (int, float))]
    return round(statistics.mean(numbers), 3) if numbers else None


def summarize(rows, reply_mode):
    critical_rows = [r for r in rows if r.get("expected_level") == "critical"]
    class_success = [r for r in rows if r.get("classification_valid")]
    replies = [r for r in rows if r.get("reply_attempted")]
    return {
        "total": len(rows),
        "classification_valid": len(class_success),
        "category_correct": sum(bool(r.get("level_correct")) for r in rows),
        "critical_correct": sum(bool(r.get("level_correct")) for r in critical_rows),
        "critical_total": len(critical_rows),
        "reply_mode": reply_mode,
        "reply_attempted": len(replies),
        "reply_schema_valid": sum(bool(r.get("reply_schema_valid")) for r in replies),
        "reply_semantic_valid": sum(bool(r.get("reply_semantic_valid")) for r in replies),
        "reply_null": sum(r.get("reply_schema_valid") and r.get("reply_value") is None for r in replies),
        "reply_nonnull": sum(r.get("reply_schema_valid") and isinstance(r.get("reply_value"), str) for r in replies),
        "reply_string_null_errors": sum(r.get("reply_semantic_error") == "string_null_instead_of_json_null" for r in replies),
        "mean_classifier_elapsed_s": describe_mean(r.get("classifier", {}).get("metrics", {}).get("elapsed_s") for r in class_success),
        "mean_reply_elapsed_s": describe_mean(r.get("reply", {}).get("metrics", {}).get("elapsed_s") for r in replies),
        "mean_total_elapsed_s": describe_mean(r.get("total_elapsed_s") for r in rows),
        "mean_classifier_prompt_tokens": describe_mean(r.get("classifier", {}).get("metrics", {}).get("prompt_tokens") for r in class_success),
        "mean_reply_prompt_tokens": describe_mean(r.get("reply", {}).get("metrics", {}).get("prompt_tokens") for r in replies),
        "mean_total_prompt_tokens": describe_mean(r.get("total_prompt_tokens") for r in rows),
        "mean_total_output_tokens": describe_mean(r.get("total_output_tokens") for r in rows),
        "mean_total_tokens": describe_mean(r.get("total_tokens") for r in rows),
        "sum_prompt_tokens": sum(r.get("total_prompt_tokens") or 0 for r in rows),
        "sum_output_tokens": sum(r.get("total_output_tokens") or 0 for r in rows),
    }


def run_stage(args, prompt, payload, schema, max_tokens, validator):
    result = {"schema_valid": False}
    try:
        raw, metrics = request_stage(args, prompt=prompt, case_data=payload,
                                     schema=schema, max_tokens=max_tokens)
        result.update(raw_answer=raw, metrics=metrics)
        parsed = json.loads(raw)
        result["parsed"] = validator(parsed)
        result["schema_valid"] = True
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def save_csv(path, rows):
    fields = [
        "id", "sender", "expected_language", "expected_level", "predicted_level",
        "classification_valid", "level_correct", "reason", "reply_attempted",
        "reply_schema_valid", "reply_semantic_valid", "reply_semantic_error",
        "suggested_reply", "classifier_elapsed_s", "reply_elapsed_s", "total_elapsed_s",
        "classifier_prompt_tokens", "reply_prompt_tokens", "total_prompt_tokens",
        "classifier_output_tokens", "reply_output_tokens", "total_output_tokens",
        "classifier_error", "reply_error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            c = row.get("classifier", {})
            r = row.get("reply", {})
            cm = c.get("metrics", {})
            rm = r.get("metrics", {})
            ans = row.get("answer", {})
            writer.writerow({
                "id": row.get("id"), "sender": row.get("sender"),
                "expected_language": row.get("expected_language"),
                "expected_level": row.get("expected_level"),
                "predicted_level": ans.get("urgency_level"),
                "classification_valid": row.get("classification_valid"),
                "level_correct": row.get("level_correct"),
                "reason": ans.get("reason"),
                "reply_attempted": row.get("reply_attempted"),
                "reply_schema_valid": row.get("reply_schema_valid"),
                "reply_semantic_valid": row.get("reply_semantic_valid"),
                "reply_semantic_error": row.get("reply_semantic_error"),
                "suggested_reply": ans.get("suggested_reply"),
                "classifier_elapsed_s": cm.get("elapsed_s"),
                "reply_elapsed_s": rm.get("elapsed_s"),
                "total_elapsed_s": row.get("total_elapsed_s"),
                "classifier_prompt_tokens": cm.get("prompt_tokens"),
                "reply_prompt_tokens": rm.get("prompt_tokens"),
                "total_prompt_tokens": row.get("total_prompt_tokens"),
                "classifier_output_tokens": cm.get("output_tokens"),
                "reply_output_tokens": rm.get("output_tokens"),
                "total_output_tokens": row.get("total_output_tokens"),
                "classifier_error": c.get("error"), "reply_error": r.get("error"),
            })


def main():
    parser = argparse.ArgumentParser(description="SmartNotify v10: separate classifier and reply generator")
    parser.add_argument("--label", required=True, help="Унікальна мітка запуску (a-z, 0-9, -, _)")
    parser.add_argument("--model", default="qwen3.5:2b-q4_K_M")
    parser.add_argument("--cases-file", type=Path, default=HERE / "cases" / "quality_10.json")
    parser.add_argument("--classifier-prompt", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--reply-prompt", type=Path, default=DEFAULT_REPLY)
    parser.add_argument("--reply-mode", choices=("all", "none", "requested"), default="all",
                        help="all: порівняння чернеток; none: лише класифікація; requested: кейси з draft_requested=true")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=220, help="Ліміт ДЛЯ КОЖНОГО етапу")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Перевірка шляхів і формату без запитів до Ollama")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("--label: тільки латиниця, цифри, - і _")
    if not (0 <= args.temperature <= 2 and args.num_ctx >= 512 and args.max_tokens >= 48):
        parser.error("Перевір temperature, num_ctx, max_tokens")

    try:
        case_bytes = args.cases_file.read_bytes()
        cases = json.loads(case_bytes.decode("utf-8-sig"))
        classifier_prompt = args.classifier_prompt.read_text(encoding="utf-8-sig")
        reply_prompt = args.reply_prompt.read_text(encoding="utf-8-sig")
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"Помилка читання файлів: {exc}")
    if not isinstance(cases, list) or not cases or not classifier_prompt.strip() or not reply_prompt.strip():
        parser.error("Промпт або набір кейсів порожній/некоректний")
    ids = []
    for case in cases:
        if (not isinstance(case, dict) or any(k not in case for k in ("id", "sender", "message", "expected_level"))
                or case["expected_level"] not in LEVELS or not isinstance(case["message"], str)
                or not isinstance(case["sender"], str)):
            parser.error("Кейси мають містити id, sender, message, expected_level")
        ids.append(str(case["id"]))
    if len(ids) != len(set(ids)):
        parser.error("ID кейсів повинні бути унікальними")

    print(f"Набір: {len(cases)} | модель: {args.model} | контекст: {args.num_ctx} | reply-mode: {args.reply_mode}")
    print(f"SHA256 тестів: {sha_bytes(case_bytes)}")
    if args.check_only:
        print("Перевірка пройшла. Запити до Ollama НЕ виконувалися.")
        return

    out = ROOT / "experiments" / "results" / f"evaluation_{args.label}.json"
    out_csv = out.with_suffix(".csv")
    if out.exists() or out_csv.exists():
        parser.error(f"Файл для --label {args.label!r} уже існує; вибери іншу мітку")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label": args.label, "model": args.model, "test_set": "custom",
        "architecture": "two_stage_classifier_then_optional_reply",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases_file": str(args.cases_file), "cases_sha256": sha_bytes(case_bytes),
        "classifier_prompt_file": str(args.classifier_prompt),
        "classifier_prompt_sha256": sha_bytes(classifier_prompt.encode("utf-8")),
        "reply_prompt_file": str(args.reply_prompt),
        "reply_prompt_sha256": sha_bytes(reply_prompt.encode("utf-8")),
        "settings": {"temperature": args.temperature, "seed": args.seed,
                     "num_ctx": args.num_ctx, "max_tokens_per_stage": args.max_tokens,
                     "think": False, "format": "json_schema", "backend": "ollama_native",
                     "reply_mode": args.reply_mode},
        "note": "Два послідовні запити. Точність категорії оцінює лише етап 1; якість чернетки потребує ручної оцінки. Holdout не запускається автоматично.",
        "warmup": {}, "cases": [], "summary": None,
    }
    if not args.no_warmup:
        print("Прогрів класифікатора й генератора (час не входить до метрик кейсів)...", flush=True)
        try:
            _, m = request_stage(args, prompt=classifier_prompt,
                    case_data={"sender": "warmup", "message": "Привіт"},
                    schema=CLASS_SCHEMA, max_tokens=args.max_tokens)
            report["warmup"]["classifier"] = m
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            report["warmup"]["classifier"] = {"error": str(exc)}
        if args.reply_mode != "none":
            try:
                _, m = request_stage(args, prompt=reply_prompt,
                        case_data={"sender": "warmup", "message": "Привіт", "urgency_level": "low"},
                        schema=REPLY_SCHEMA, max_tokens=args.max_tokens)
                report["warmup"]["reply"] = m
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                report["warmup"]["reply"] = {"error": str(exc)}

    for case in cases:
        row = dict(case)
        classifier = run_stage(args, classifier_prompt,
            {"sender": case["sender"], "message": case["message"]},
            CLASS_SCHEMA, args.max_tokens, validate_classifier)
        row["classifier"] = classifier
        row["classification_valid"] = classifier["schema_valid"]
        class_value = classifier.get("parsed", {}) if classifier["schema_valid"] else {}
        row["level_correct"] = class_value.get("urgency_level") == case["expected_level"]
        row["answer"] = {"urgency_level": class_value.get("urgency_level"),
                         "reason": class_value.get("reason"), "suggested_reply": None}
        if args.reply_mode == "all":
            should_reply = True
        elif args.reply_mode == "requested":
            should_reply = case.get("draft_requested") is True
        else:
            should_reply = False
        # No draft call when classification failed: preserve error and separate metrics.
        should_reply = should_reply and classifier["schema_valid"]
        row["reply_attempted"] = should_reply
        row["reply_schema_valid"] = None
        row["reply_semantic_valid"] = None
        row["reply_semantic_error"] = None
        if should_reply:
            result = run_stage(args, reply_prompt,
                {"sender": case["sender"], "message": case["message"],
                 "urgency_level": class_value["urgency_level"]},
                REPLY_SCHEMA, args.max_tokens, validate_reply)
            reply_value, semantic_error = result.get("parsed", (None, None))
            row["reply"] = result
            row["reply_schema_valid"] = result["schema_valid"]
            row["reply_semantic_error"] = semantic_error
            row["reply_semantic_valid"] = result["schema_valid"] and semantic_error is None
            if result["schema_valid"]:
                row["reply_value"] = reply_value
                row["answer"]["suggested_reply"] = reply_value
        else:
            row["reply"] = {"skipped": True, "reason": "not_requested_or_classification_failed"}

        stage_metrics = [stage.get("metrics", {}) for stage in (row["classifier"], row["reply"])]
        row["total_elapsed_s"] = round(sum(m.get("elapsed_s") or 0 for m in stage_metrics), 3)
        row["total_prompt_tokens"] = sum(m.get("prompt_tokens") or 0 for m in stage_metrics)
        row["total_output_tokens"] = sum(m.get("output_tokens") or 0 for m in stage_metrics)
        row["total_tokens"] = row["total_prompt_tokens"] + row["total_output_tokens"]
        report["cases"].append(row)
        report["summary"] = summarize(report["cases"], args.reply_mode)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        predicted = class_value.get("urgency_level") or "ERROR"
        rm = row.get("reply", {}).get("metrics", {})
        print(f"{case['id']}: {predicted:<8} / {case['expected_level']:<8} "
              f"| classifier={classifier.get('metrics', {}).get('elapsed_s')}s "
              f"| reply={rm.get('elapsed_s', 'skip')}s | total={row['total_elapsed_s']}s", flush=True)

    save_csv(out_csv, report["cases"])
    s = report["summary"]
    print(f"\nПідсумок: {s['category_correct']}/{s['total']} категорій; "
          f"critical {s['critical_correct']}/{s['critical_total']}; "
          f"reply JSON {s['reply_schema_valid']}/{s['reply_attempted']}; "
          f"draft null {s['reply_null']}; mean total {s['mean_total_elapsed_s']} s")
    print("JSON:", out, "\nCSV: ", out_csv)


if __name__ == "__main__":
    main()
