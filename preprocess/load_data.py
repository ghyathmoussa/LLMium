from datasets import load_dataset
import json
from tqdm import tqdm
import gc
import os

# OPTIMIZATION 1: Use streaming mode - doesn't load entire dataset into memory
dataset = load_dataset(
    "oscar-corpus/OSCAR-2301", 
    "ar", 
    language="ar", 
    token=os.getenv("HF_TOKEN"), 
    cache_dir="../data/",
    streaming=True  # CRITICAL: Stream data instead of loading all at once
)

print(f"Dataset type: {type(dataset['train'])}")

# OPTIMIZATION 2: Process and write directly to JSONL without storing in memory
# OPTIMIZATION 3: Use batching to process in chunks
batch_size = 1000  # Process 1000 records at a time
max_records = 500000  # Limit to first 1M records
record_count = 0
batch_count = 0

output_file = "../data/oscar_ar.jsonl"  # Use .jsonl extension for clarity

with open(output_file, "w", encoding="utf-8") as f:
    batch = []
    
    for item in tqdm(dataset["train"], desc="Processing dataset", total=max_records):
        try:
            # Extract only the fields you need
            obj = {
                "text": item["text"],
                "categories": item.get("meta", {}).get("categories", []) if item.get("meta") else []
            }
            
            # Write directly to file (no accumulation in memory)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
            record_count += 1
            batch.append(obj)
            
            # Clear batch periodically and force garbage collection
            if len(batch) >= batch_size:
                batch.clear()
                batch_count += 1
                gc.collect()  # Free up memory
            
            # Stop after reaching max_records
            if record_count >= max_records:
                print(f"\nReached limit of {max_records:,} records. Stopping...")
                break
                
        except Exception as e:
            print(f"Error processing record {record_count}: {e}")
            continue

print(f"\nProcessing complete!")
print(f"Total records processed: {record_count:,}")
print(f"Output file: {output_file}")

# Note: To change the limit, modify the max_records variable at the top
# To process all records, set max_records to float('inf')
# To get random sample instead of first N records, use:
# dataset_sample = dataset["train"].shuffle(seed=42).take(max_records)