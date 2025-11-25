import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            ents = []
            # Post-process spans with lightweight validators to improve precision
            # Spoken digit words that count as numeric content
            SPOKEN_DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "oh", "double", "triple"}

            def count_digit_content(span_text: str) -> int:
                """Count digits + spoken digit words"""
                digits = len(re.findall(r"\d", span_text))
                words = span_text.lower().split()
                spoken = sum(1 for w in words if w in SPOKEN_DIGITS)
                return digits + spoken

            def is_credit_card(span_text: str) -> bool:
                # Credit cards: 13-19 digit content (actual digits or spoken words)
                count = count_digit_content(span_text)
                return 10 <= count <= 25  # lenient range for spoken forms with "double"/"triple"

            def is_phone(span_text: str) -> bool:
                # Phones: 7-15 digit content
                count = count_digit_content(span_text)
                return 7 <= count <= 18

            def is_email(span_text: str) -> bool:
                # Accept raw emails and spoken 'at'/'dot' forms
                if "@" in span_text:
                    return True
                if re.search(r"\b(at)\b", span_text.lower()):
                    return True
                if " dot " in span_text.lower() or "." in span_text:
                    return bool(re.search(r"[a-zA-Z]", span_text))
                return False

            def is_person_name(span_text: str) -> bool:
                # Names: no numeric digits, 2-50 chars, has letters
                # Allow spoken words that aren't digit words
                if re.search(r"\d", span_text):
                    return False
                words = span_text.lower().split()
                # If most words are spoken digits, it's not a name
                digit_word_count = sum(1 for w in words if w in SPOKEN_DIGITS)
                if len(words) > 0 and digit_word_count / len(words) > 0.5:
                    return False
                stripped = span_text.strip()
                if len(stripped) < 2 or len(stripped) > 50:
                    return False
                return bool(re.search(r"[a-zA-Z]", stripped))

            for s, e, lab in spans:
                span_text = text[int(s):int(e)]
                keep = True
                if lab == "CREDIT_CARD":
                    keep = is_credit_card(span_text)
                elif lab == "PHONE":
                    keep = is_phone(span_text)
                elif lab == "EMAIL":
                    keep = is_email(span_text)
                elif lab == "PERSON_NAME":
                    keep = is_person_name(span_text)

                if not keep:
                    continue

                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
