import json

with open("../data/english/puffin.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

print(f"Total data: {len(data)}")

all_data = []

for item in data:
    conversations = item["instruction"]
    
    for i in range(0, len(conversations), 2):
        obj = {
            "instruction": conversations[i]["value"],
            "output": conversations[i+1]["value"],
        }
        all_data.append(obj)


with open("../data/english/puffin_processed.jsonl", "w") as f:
    for item in all_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")