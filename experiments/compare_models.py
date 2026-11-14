"""Порівняння чотирьох сирих JSON-запусків SmartNotify v10 (без фільтрації / корекції)."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from pathlib import Path

LEVELS = ('low', 'medium', 'high', 'critical')
FIELDS = [
    'model', 'id', 'sender', 'message', 'expected_level', 'predicted_level',
    'classification_valid', 'level_correct', 'reason', 'reply_attempted',
    'reply_schema_valid', 'reply_semantic_valid', 'suggested_reply', 'reply_value_type',
    'elapsed_s', 'classifier_error', 'reply_error',
    'reason_correct_yes_no', 'reason_natural_uk_yes_no',
    'reply_natural_uk_yes_no_na', 'reply_roles_correct_yes_no_na',
    'reply_no_invented_facts_yes_no_na', 'reply_helpful_yes_no_na', 'reply_null_appropriate_yes_no_na',
    'review_notes'
]


def readable(v):
    return 'null' if v is None else str(v)


def load(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or not isinstance(data.get('cases'), list):
        raise ValueError(f'{path}: не схожий на JSON-результат split runner')
    if not data.get('model') or len(data['cases']) != 10:
        raise ValueError(f'{path}: очікується 10 кейсів і назва моделі')
    return data


def percentile_median(rows, field):
    values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
    return round(statistics.median(values), 3) if values else None


def main():
    parser = argparse.ArgumentParser(description='Незмінені результати SmartNotify: автоматичні категорії + таблиця для ручної оцінки')
    parser.add_argument('--results', nargs='+', type=Path, required=True, help='Чотири JSON від run_evaluation.py')
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/results/recomputed'))
    args = parser.parse_args()
    if len(args.results) != 4:
        parser.error('Вкажіть рівно 4 файли, по одному на кожну модель')
    data = [load(p) for p in args.results]
    names = [d['model'] for d in data]
    if len(set(names)) != len(names):
        parser.error('Моделі мають бути різними; дублікати не можна порівнювати')
    for attr in ('cases_sha256', 'classifier_prompt_sha256', 'reply_prompt_sha256'):
        vals = [d.get(attr) for d in data]
        if not all(vals) or len(set(vals)) != 1:
            parser.error(f'{attr} відрізняється або відсутній. Це НЕ однаковий тест')
    settings = ['temperature', 'seed', 'num_ctx', 'max_tokens_per_stage', 'think', 'format', 'reply_mode']
    for attr in settings:
        vals = [d.get('settings', {}).get(attr) for d in data]
        if len(set(map(str, vals))) != 1:
            parser.error(f'Налаштування {attr} не збігаються')
    by_model = {}
    for d in data:
        case_map = {str(c.get('id')): c for c in d['cases']}
        if len(case_map) != 10:
            parser.error(f'{d["model"]}: дублікати id кейсів')
        by_model[d['model']] = case_map
    ids = sorted(next(iter(by_model.values())))
    for m, cases in by_model.items():
        if sorted(cases) != ids:
            parser.error(f'{m}: відрізняються id кейсів')
    # Метадані очікуваних відповідей звіряються по кожному кейсу.
    for case_id in ids:
        ref = next(iter(by_model.values()))[case_id]
        for m, cases in by_model.items():
            case = cases[case_id]
            for attr in ('sender', 'message', 'expected_level'):
                if case.get(attr) != ref.get(attr):
                    parser.error(f'{m} {case_id}: відрізняється {attr}')
    out_rows = []
    summaries = []
    for d in data:
        model = d['model']
        raw_cases = [by_model[model][i] for i in ids]
        for c in raw_cases:
            answer = c.get('answer') or {}
            classifier = c.get('classifier') or {}
            reply = c.get('reply') or {}
            out_rows.append({
                'model': model, 'id': c['id'], 'sender': c['sender'], 'message': c['message'],
                'expected_level': c['expected_level'],
                'predicted_level': answer.get('urgency_level') or '',
                'classification_valid': c.get('classification_valid'),
                'level_correct': c.get('level_correct'), 'reason': answer.get('reason') or '',
                'reply_attempted': c.get('reply_attempted'),
                'reply_schema_valid': c.get('reply_schema_valid'),
                'reply_semantic_valid': c.get('reply_semantic_valid'),
                'suggested_reply': readable(answer.get('suggested_reply')),
                'reply_value_type': ('json_null' if answer.get('suggested_reply') is None else
                                     'literal_string_null' if str(answer.get('suggested_reply')).strip().casefold() == 'null' else 'string'),
                'elapsed_s': c.get('total_elapsed_s'),
                'classifier_error': classifier.get('error') or '',
                'reply_error': reply.get('error') or '',
            })
        correct = sum(c.get('level_correct') is True for c in raw_cases)
        crit = [c for c in raw_cases if c.get('expected_level') == 'critical']
        medium_high = [c for c in raw_cases if c.get('expected_level') in ('medium', 'high')]
        summaries.append({
            'model': model,
            'correct_of_10': correct,
            'critical_correct_of_2': f'{sum(c.get("level_correct") is True for c in crit)}/{len(crit)}',
            'medium_high_correct_of_4': f'{sum(c.get("level_correct") is True for c in medium_high)}/{len(medium_high)}',
            'class_json_valid_of_10': sum(c.get('classification_valid') is True for c in raw_cases),
            'reply_json_valid_of_10': sum(c.get('reply_schema_valid') is True for c in raw_cases),
            'reply_json_null': sum((c.get('reply_schema_valid') is True and c.get('answer', {}).get('suggested_reply') is None) for c in raw_cases),
            'median_total_elapsed_s': percentile_median(raw_cases, 'total_elapsed_s'),
            'next_action': 'Ручна перевірка мови, ролей, фактів, доречності — рішення після неї',
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'quality_manual_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    with (args.output_dir / 'quality_auto_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    md = [
        '# SmartNotify — 10 українських кейсів / чотири моделі', '',
        'Це порівняння **сирих виходів LLM**. Нічого не переписано, не заблоковано і не виправлено.',
        '', '| Модель | Категорії | Critical | Medium+High | Класифікатор JSON | Генератор JSON | Медіана часу, с |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for x in summaries:
        md.append(f'| {x["model"]} | {x["correct_of_10"]}/10 | {x["critical_correct_of_2"]} | {x["medium_high_correct_of_4"]} | {x["class_json_valid_of_10"]}/10 | {x["reply_json_valid_of_10"]}/10 | {x["median_total_elapsed_s"]} |')
    md.extend(['', '## Важливо', '',
               '- 10 тестів — мала вибірка для надійного узагальнення; це початковий скринінг для ЛР4.',
               '- Набір Q01–Q10 походить із development-набору. Це не незалежний holdout.',
               '- «reply_semantic_valid» у runner **не є оцінкою змісту**: лише тип / формальна структура.',
               '- Спершу заповнити `quality_manual_review.csv`, потім робити висновок про якість.',
               '- `null` не дорівнює граматичній помилці; оцінюйте доречність відсутності чернетки за сценарієм.',
               '- Всі чотири моделі отримали однаковий промпт і параметри, але промпт раніше оптимізували на Qwen3.5. Це обмеження справедливості порівняння.',
               ''])
    (args.output_dir / 'quality_summary.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))
    print('Ручна оцінка:', args.output_dir / 'quality_manual_review.csv')
    print('CSV з метриками:', args.output_dir / 'quality_auto_summary.csv')


if __name__ == '__main__':
    main()
