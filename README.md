# PII NER Assignment Skeleton

This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

## Evaluate

```bash
# Dev set
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

# (Optional) stress test set
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

Your task in the assignment is to modify the model and training code to improve entity and PII detection quality while keeping **p95 latency below ~20 ms** per utterance (batch size 1, on a reasonably modern CPU).

---

## What I Did

1. **Fixed the dev set** — Original had only 10 samples. Created proper 80/20 split (688 train / 172 dev).

2. **Upgraded model** — Switched from `distilbert-base-uncased` to `distilroberta-base` for better noisy text handling.

3. **Added post-processing validators** — Lightweight filters in `predict.py` to improve precision on CREDIT_CARD, PHONE, EMAIL, PERSON_NAME.

4. **Data augmentation** — Created `augment_data.py` to generate spoken-digit variations of credit cards and phones (e.g., "4242" → "four two four two") since stress test uses STT-style spoken numbers.

### Results

| Metric | Baseline | Final |
|--------|----------|-------|
| Stress PII F1 | 0.631 | **0.768** |
| Stress Macro-F1 | 0.519 | **0.689** |
| p95 Latency | ~8ms | ~8ms ✅ |

📄 **See [`changes.md`](./changes.md) for detailed breakdown of all changes and phase-by-phase results.**

## Reproducing
Just run `./04.run.sh &> >(tee out.log)` for reproducing the final state of this project
