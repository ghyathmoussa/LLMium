from datasets import load_dataset
import json
from tqdm import tqdm
import gc
import os

# OPTIMIZATION 1: Use streaming mode - doesn't load entire dataset into memory
HF_TOKEN=""
dataset = load_dataset(
    "nampdn-ai/tiny-codes", 
    # "ar", 
    # language="ar", 
    token=HF_TOKEN, 
    cache_dir="../data/",
    streaming=True  # CRITICAL: Stream data instead of loading all at once
)

print(f"Dataset type: {type(dataset['train'])}")

# OPTIMIZATION 2: Process and write directly to JSONL without storing in memory
# OPTIMIZATION 3: Use batching to process in chunks
# batch_size = 1000  # Process 1000 records at a time
# max_records = 500000  # Limit to first 1M records
record_count = 0
# batch_count = 0

output_file = "../data/english/tiny-codes.jsonl"  # Use .jsonl extension for clarity

all_data = []

for item in tqdm(dataset["train"], desc="Processing dataset"):
    try:
        # Extract only the fields you need
        obj = {
            "instruction": item["prompt"],
            "output": item["response"],
            "programming_language": item["programming_language"]
        }
        
        # Write directly to file (no accumulation in memory)
        # f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
        record_count += 1
        all_data.append(obj)
        # # Stop after reaching max_records
        # if record_count >= max_records:
        #     print(f"\nReached limit of {max_records:,} records. Stopping...")
        #     break
            
    except Exception as e:
        print(f"Error processing record {record_count}: {e}")
        continue

with open(output_file, 'w', encoding='utf-8') as outfile:
    for entry in all_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"\nProcessing complete!")
print(f"Total records processed: {record_count:,}")
print(f"Output file: {output_file}")

# Note: To change the limit, modify the max_records variable at the top
# To process all records, set max_records to float('inf')
# To get random sample instead of first N records, use:
# dataset_sample = dataset["train"].shuffle(seed=42).take(max_records)