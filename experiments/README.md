# Відтворювані експерименти SmartNotify

Мета цієї теки — зберегти **ключові докази вибору моделі й оптимізації**, а не весь щоденник перебору промптів. `results/*.json` — незмінені сирі файли з локальних запусків; `*.csv` — таблиці, згенеровані із цих файлів.

## 1. Порівняння моделей на одному наборі

- `cases/quality_10.json` — 10 українських повідомлень. `expected_level` і `manual_goal` потрібні **лише для оцінки**, модель не бачить їх.
- `prompts/classifier_baseline.txt`, `prompts/reply_baseline.txt` — однакові контрольні промпти для **чотирьох** моделей.
- `results/model_qwen3_0_6b.json`, `model_qwen3_1_7b.json`, `model_qwen35_2b.json`, `model_qwen3_4b.json` — повні результати двох етапів.
- `results/model_comparison.csv` — автоматична зведена таблиця; `results/model_cases.csv` — 40 рядків із сирими відповідями для ручної перевірки.

Показник правильних категорій: **5/10, 7/10, 8/10, 9/10** відповідно. На 10 development-кейсах ці цифри не є статистично надійною оцінкою для реального використання.

## 2. Оптимізація класифікатора

- `results/classifier_compact.json` — Qwen3.5-2B з `app/prompts/classifier.txt`; порівняно з `results/model_qwen35_2b.json` (старий контрольний промпт) на тих самих 10 повідомленнях. У першому експерименті додатково генерувалася чернетка, тому для порівняння часу беріть **лише метрики етапу класифікації**.
- `results/classifier_comparison.csv` — порівняння правильності, пояснень, часу та вхідних токенів по кейсах. Обидва варіанти — 8/10; у компактного менше вхідних токенів, але це не доводить кращу точність.

## 3. Генерація чернеток

- `prompts/reply_candidate.txt` — попередній компактний генератор.
- `app/prompts/reply.txt` — обраний генератор після коригування ролей і екстрених випадків.
- `results/reply_ab_final.json`, `reply_comparison.csv` — парний тест на **кешованих, однакових для обох промптів результатах класифікатора**. Не вважайте структурно коректний JSON доказом правильності чернетки.

## Назви й історичні метадані

Файли перейменовано для читабельності, але поля `label`, `source_path`, `classifier_prompt_file`, `reply_prompt_file`, `prompt_A.path` і `prompt_B.path` у сирих JSON **не переписувалися**, щоб залишити оригінальний журнал запуску. Мапа старих назв на нові:

| Історична назва | Поточний файл |
|---|---|
| `system_v10_classifier.txt` | `experiments/prompts/classifier_baseline.txt` |
| `system_v10_reply.txt` | `experiments/prompts/reply_baseline.txt` |
| `system_v11_classifier_en.txt` | `app/prompts/classifier.txt` |
| `system_v11_reply_en.txt` | `experiments/prompts/reply_candidate.txt` |
| `system_v11_1_reply_en.txt` | `app/prompts/reply.txt` |
| `lr4_smartnotify_quality_10_uk.json` | `experiments/cases/quality_10.json` |

SHA-256 в історичних JSON відповідають незміненим текстам промптів та кейсів. Керовані параметри: temperature=0, seed=1, context=4096, max_tokens=220, think=false.

## Повторення запуску (вимагає Ollama)

Використовується окремий тестовий виконавець із `/api/chat` (не UI), щоби збирати фактичні токени та таймінги Ollama. З кореня репозиторію:

```powershell
python .\experiments\run_evaluation.py --model qwen3.5:2b-q4_K_M --label new_quality --reply-mode all
```

Для відтворення **старого** контрольного набору промптів задайте їх явно:

```powershell
python .\experiments\run_evaluation.py --model qwen3.5:2b-q4_K_M --label baseline_recheck --cases-file .\experiments\cases\quality_10.json --classifier-prompt .\experiments\prompts\classifier_baseline.txt --reply-prompt .\experiments\prompts\reply_baseline.txt --reply-mode all
```

Для A/B на збереженій класифікації:

```powershell
python .\experiments\compare_replies.py --source-results .\experiments\results\model_qwen35_2b.json --model qwen3.5:2b-q4_K_M --label new_reply_ab
```

Для автоматичної таблиці чотирьох моделей:

```powershell
python .\experiments\compare_models.py --results .\experiments\results\model_qwen3_0_6b.json .\experiments\results\model_qwen3_1_7b.json .\experiments\results\model_qwen35_2b.json .\experiments\results\model_qwen3_4b.json
```

Результати нових запусків не перезаписують історичні JSON. Ручну оцінку української, ролей, фактів і безпечності не можна автоматично заміняти перевіркою валідності JSON.
