import json
import random
import os

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    random.seed(42)
    
    # Load existing data
    train_data = load_jsonl('data/train.jsonl')
    dev_data = load_jsonl('data/dev.jsonl')
    
    # Combine and shuffle
    all_data = train_data + dev_data
    random.shuffle(all_data)
    
    # 80/20 split
    split_idx = int(len(all_data) * 0.8)
    new_train = all_data[:split_idx]
    new_dev = all_data[split_idx:]
    
    print(f"Total examples: {len(all_data)}")
    print(f"New Train: {len(new_train)}")
    print(f"New Dev: {len(new_dev)}")
    
    save_jsonl(new_train, 'data/new_train.jsonl')
    save_jsonl(new_dev, 'data/new_dev.jsonl')

if __name__ == "__main__":
    main()
