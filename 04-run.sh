export TOKENIZERS_PARALLELISM=true
set -e

python src/train.py \
  --model_name distilroberta-base \
  --train data/new_train.jsonl \
  --dev data/new_dev.jsonl \
  --epochs 5 \
  --out_dir out

python src/predict.py \
  --model_dir out \
  --input data/new_dev.jsonl \
  --output out/dev_pred.json

python src/eval_span_f1.py \
  --gold data/new_dev.jsonl \
  --pred out/dev_pred.json

python src/measure_latency.py \
  --model_dir out \
  --input data/new_dev.jsonl \
  --runs 50

python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json

python src/measure_latency.py \
  --model_dir out \
  --input data/stress.jsonl \
  --runs 50