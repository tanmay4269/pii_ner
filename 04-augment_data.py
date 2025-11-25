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
    entities = obj.get('entities', [])
    
    # Sort by start position (ascending) to process left-to-right
    sorted_entities = sorted(enumerate(entities), key=lambda x: x[1]['start'])
    
    new_text = text
    new_entities = [ent.copy() for ent in entities]
    cumulative_offset = 0
    
    for orig_idx, ent in sorted_entities:
        label = ent['label']
        orig_start, orig_end = ent['start'], ent['end']
        
        # Adjust for previous augmentations
        adj_start = orig_start + cumulative_offset
        adj_end = orig_end + cumulative_offset
        span_text = new_text[adj_start:adj_end]
        
        if label in ('CREDIT_CARD', 'PHONE') and re.search(r'\d', span_text):
            spoken = digits_to_spoken(span_text)
            new_text = new_text[:adj_start] + spoken + new_text[adj_end:]
            
            length_diff = len(spoken) - (orig_end - orig_start)
            
            # Update this entity's positions
            new_entities[orig_idx]['start'] = adj_start
            new_entities[orig_idx]['end'] = adj_start + len(spoken)
            
            # Update all entities that come after this one
            for other_idx, other_ent in enumerate(entities):
                if other_ent['start'] > orig_start:
                    new_entities[other_idx]['start'] += length_diff
                    new_entities[other_idx]['end'] += length_diff
            
            cumulative_offset += length_diff
        else:
            # Still need to apply cumulative offset
            new_entities[orig_idx]['start'] = adj_start
            new_entities[orig_idx]['end'] = adj_end
    
    return {
        'id': obj['id'] + '_aug',
        'text': new_text,
        'entities': new_entities
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