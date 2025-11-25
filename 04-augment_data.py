"""
Augment training data by converting numeric entities to spoken digit forms.
"""
import json
import random
import re

DIGIT_WORDS = {
    '0': ['zero', 'oh'],
    '1': ['one'],
    '2': ['two'],
    '3': ['three'],
    '4': ['four'],
    '5': ['five'],
    '6': ['six'],
    '7': ['seven'],
    '8': ['eight'],
    '9': ['nine'],
}

def digits_to_spoken(text: str) -> str:
    result = []
    for char in text:
        if char.isdigit():
            result.append(random.choice(DIGIT_WORDS[char]))
        elif char in ' -':
            if random.random() > 0.5:
                result.append('')
        else:
            result.append(char)
    return ' '.join(w for w in result if w).strip()

def augment_utterance(obj: dict) -> dict:
    text = obj['text']
    entities = sorted(obj.get('entities', []), key=lambda x: x['start'], reverse=True)
    
    new_text = text
    new_entities = []
    
    for ent in entities:
        label = ent['label']
        start, end = ent['start'], ent['end']
        span_text = text[start:end]
        
        if label in ('CREDIT_CARD', 'PHONE') and re.search(r'\d', span_text):
            spoken = digits_to_spoken(span_text)
            new_text = new_text[:start] + spoken + new_text[end:]
            new_entities.append({'start': start, 'end': start + len(spoken), 'label': label})
        else:
            new_entities.append(ent.copy())
    
    return {
        'id': obj['id'] + '_aug',
        'text': new_text,
        'entities': list(reversed(new_entities))
    }

def main():
    random.seed(42)
    
    with open('data/new_train.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f]
    
    augmented = []
    for obj in train_data:
        labels = [e['label'] for e in obj.get('entities', [])]
        if 'CREDIT_CARD' in labels or 'PHONE' in labels:
            try:
                augmented.append(augment_utterance(obj))
            except:
                pass
    
    print(f"Original: {len(train_data)}, Augmented: {len(augmented)}")
    
    combined = train_data + augmented
    random.shuffle(combined)
    
    with open('data/aug_train.jsonl', 'w') as f:
        for item in combined:
            f.write(json.dumps(item) + '\n')
    
    print(f"Total: {len(combined)} -> data/aug_train.jsonl")

if __name__ == "__main__":
    main()