# Changes Made
(AI Generated report but reviewed by me)

---

# Change 1: Data Split Script
**File:** `split_data.py` (new file)

**Why:** The original `dev.jsonl` only had 10 samples, which caused the model to show perfect 100% scores on dev but terrible generalization on stress test. We needed a proper validation set.

**What I did:**
- Created a new Python script that loads both `train.jsonl` (851 samples) and the tiny `dev.jsonl` (10 samples)
- Combined them into a single pool of 861 utterances
- Shuffled with a fixed random seed (42) for reproducibility
- Split 80/20 to create `new_train.jsonl` (688 samples) and `new_dev.jsonl` (172 samples)
- This gives us a meaningful validation set to detect overfitting

---

# Change 2: Post-Processing Validators in Prediction
**File:** `src/predict.py`

**Why:** The baseline model was over-predicting entities, especially on stress test. For example, it was tagging random text as CREDIT_CARD or PHONE with very low precision (0.256 and 0.105 respectively). We needed to filter out obvious false positives.

**What I changed:**

1. **Added `import re`** at the top of the file for regex operations

2. **Added a set of spoken digit words** — created a constant `SPOKEN_DIGITS` containing words like "zero", "one", "two", ... "nine", plus "oh", "double", "triple" to handle STT transcripts where numbers are spoken

3. **Created a helper function `count_digit_content()`** — counts both actual digit characters AND spoken digit words in a span. This is crucial because stress test has things like "three four zero three eight one five four..." which have no actual digits but represent numbers

4. **Added `is_credit_card()` validator:**
   - Counts digit content (actual + spoken)
   - Requires 10-25 digit-equivalent content (lenient range to handle "double" and "triple")
   - Originally tried Luhn checksum but that failed on spoken forms

5. **Added `is_phone()` validator:**
   - Counts digit content (actual + spoken)  
   - Requires 7-18 digit-equivalent content

6. **Added `is_email()` validator:**
   - Accepts spans with `@` symbol
   - Accepts spoken forms like "something at something"
   - Accepts "dot" patterns with alphabetic content

7. **Added `is_person_name()` validator:**
   - Rejects spans containing actual digit characters
   - Rejects spans where more than 50% of words are spoken digits (to avoid tagging "three four five" as a name)
   - Requires 2-50 characters with at least one letter

8. **Modified the span collection loop** — before appending each entity to results, I now check if it passes the appropriate validator. If `keep` is False, I skip adding that entity.

**Outcome:** This improved precision but initially hurt recall badly because the validators were too strict (especially Luhn check which doesn't work on spoken digits).

---

# Change 3: Updated Run Script
**File:** `run.sh`

**What I changed:**

1. **Switched model** from `distilbert-base-uncased` to `distilroberta-base`
   - RoBERTa generally handles noisy text better
   - Similar size and latency profile

2. **Increased epochs** from 3 to 5
   - More training helps the model converge better on the smaller training set

3. **Updated data paths** to use the new split files:
   - `--train data/new_train.jsonl` instead of `data/train.jsonl`
   - `--dev data/new_dev.jsonl` instead of `data/dev.jsonl`
   - Evaluation still runs on both `new_dev.jsonl` and `stress.jsonl`

4. **Added latency measurement** for both dev and stress sets using `measure_latency.py`

---

# Change 4: Data Augmentation Script
**File:** `augment_data.py` (new file)

**Why:** The stress test contains spoken digit forms like "three four zero three eight one five four seven one nine four one six three one" for credit cards, but the training data mostly has numeric forms like "3403815471941631". The model never learned to recognize spoken digits as entity content.

**What I did:**
- Created a mapping of digits to spoken words (e.g., '0' → ['zero', 'oh'], '1' → ['one'], etc.)
- For each training sample that contains CREDIT_CARD or PHONE entities:
  - Extract the entity span text
  - Convert each digit to a randomly chosen spoken word
  - Replace the span in the text with the spoken version
  - Adjust entity offsets accordingly
  - Save as a new sample with "_aug" suffix on the ID
- Combined original training data with augmented samples
- Shuffled and saved to `data/aug_train.jsonl`

**Result:** This roughly doubles the training samples that have credit cards and phones, with half being numeric and half being spoken forms.

---

# Summary of Results

## Baseline (Their Boilerplate Codebase)
```
Dev (10 utterances):
  Macro-F1: 1.000
  PII-only: P=1.000 R=1.000 F1=1.000

Stress (100 utterances):
  CITY            P=1.000 R=1.000 F1=1.000
  CREDIT_CARD     P=0.000 R=0.000 F1=0.000
  DATE            P=1.000 R=1.000 F1=1.000
  EMAIL           P=0.125 R=0.125 F1=0.125
  PERSON_NAME     P=0.325 R=1.000 F1=0.491
  PHONE           P=0.333 R=1.000 F1=0.500
  
  Macro-F1: 0.519
  PII-only: P=0.511 R=0.825 F1=0.631
```

---

## Phase 1: Improved Model + Longer Training + Proper Dev Split
- Switched to `distilroberta-base`
- 5 epochs instead of 3
- Created proper 172-sample dev set

```
Dev (172 utterances):
  Macro-F1: 1.000
  PII-only: P=1.000 R=1.000 F1=1.000
  Latency: p50=7.66ms, p95=8.21ms

Stress (100 utterances):
  CITY            P=1.000 R=1.000 F1=1.000
  CREDIT_CARD     P=0.256 R=0.550 F1=0.349
  DATE            P=1.000 R=1.000 F1=1.000
  EMAIL           P=1.000 R=1.000 F1=1.000
  PERSON_NAME     P=0.333 R=1.000 F1=0.500
  PHONE           P=0.105 R=0.225 F1=0.143
  
  Macro-F1: 0.665
  PII-only: P=0.446 R=0.830 F1=0.580
  Latency: p50=7.70ms, p95=7.98ms
```

---

## Phase 2: Added Post-Processing Validators (Too Strict)
- Added Luhn checksum for credit cards (failed on spoken digits)
- Added digit count validators for phone/credit card

```
Dev (172 utterances):
  CREDIT_CARD     P=1.000 R=0.045 F1=0.087  ← Validators too strict!
  PHONE           P=1.000 R=0.593 F1=0.745
  
  Macro-F1: 0.833
  PII-only: P=1.000 R=0.878 F1=0.935
  Latency: p50=7.71ms, p95=8.59ms

Stress (100 utterances):
  CITY            P=1.000 R=1.000 F1=1.000
  CREDIT_CARD     P=0.000 R=0.000 F1=0.000  ← Completely broken!
  DATE            P=1.000 R=1.000 F1=1.000
  EMAIL           P=0.925 R=0.925 F1=0.925
  PERSON_NAME     P=0.328 R=1.000 F1=0.494
  PHONE           P=0.000 R=0.000 F1=0.000  ← Completely broken!
  
  Macro-F1: 0.570
  PII-only: P=0.547 R=0.585 F1=0.565
  Latency: p50=7.77ms, p95=8.04ms
```

---

## Phase 3: Relaxed Validators + Data Augmentation
- Removed Luhn checksum (doesn't work on spoken digits)
- Changed validators to count spoken digit words ("three", "four", etc.)
- Added `augment_data.py` to create spoken-digit versions of training samples

```
Dev (172 utterances):
  All entities: P=1.000 R=1.000 F1=1.000
  
  Macro-F1: 1.000
  PII-only: P=1.000 R=1.000 F1=1.000
  Latency: p50=7.68ms, p95=8.05ms

Stress (100 utterances):
  CITY            P=1.000 R=1.000 F1=1.000
  CREDIT_CARD     P=0.513 R=0.500 F1=0.506  ← Recovered!
  DATE            P=1.000 R=1.000 F1=1.000
  EMAIL           P=0.625 R=0.625 F1=0.625
  PERSON_NAME     P=0.333 R=1.000 F1=0.500
  PHONE           P=0.500 R=0.500 F1=0.500  ← Recovered!
  
  Macro-F1: 0.689
  PII-only: P=0.659 R=0.920 F1=0.768
  Latency: p50=7.69ms, p95=8.04ms
```

---

## Final Comparison Table

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 (Final) |
|--------|----------|---------|---------|-----------------|
| Stress Macro-F1 | 0.519 | 0.665 | 0.570 | **0.689** |
| Stress PII Precision | 0.511 | 0.446 | 0.547 | **0.659** |
| Stress PII Recall | 0.825 | 0.830 | 0.585 | **0.920** |
| Stress PII F1 | 0.631 | 0.580 | 0.565 | **0.768** |
| CREDIT_CARD F1 | 0.000 | 0.349 | 0.000 | **0.506** |
| PHONE F1 | 0.500 | 0.143 | 0.000 | **0.500** |
| p95 Latency | ~8ms | ~8ms | ~8ms | **~8ms** |

**Key insight:** The validators help filter false positives but the real fix needs to come from training data augmentation with spoken digit forms. The model simply hasn't seen enough examples of "three four zero three..." style credit cards to recognize them.
