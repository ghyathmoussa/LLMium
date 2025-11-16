#!/usr/bin/env python3
"""
Example scripts demonstrating reward model usage.

These examples show how to use the reward model system for evaluating
and filtering synthetic Q&A pairs in different scenarios.
"""

import json
import os
from typing import List, Dict
from models.reward_model import RewardModelAPI, RewardModelEvaluator, create_reward_evaluator
from utils.reward_functions import (
    answer_length_reward_func,
    language_quality_reward_func,
    composite_reward_func,
    answer_relevance_reward_func,
)


def example1_basic_api_scoring():
    """Example 1: Basic scoring with RewardModelAPI"""
    print("=" * 80)
    print("Example 1: Basic Scoring with RewardModelAPI")
    print("=" * 80)
    
    # Initialize the reward model
    reward_model = RewardModelAPI(
        language="multilingual",
        api_type="huggingface",
        api_key=os.environ.get("HF_API_TOKEN")
    )
    
    # Score a single answer
    question = "What is the capital of France?"
    answer = "Paris is the capital of France."
    
    score = reward_model.score_answer_sync(question, answer)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"Score: {score:.3f}\n")


def example2_batch_scoring():
    """Example 2: Batch scoring multiple Q&A pairs"""
    print("=" * 80)
    print("Example 2: Batch Scoring Multiple Q&A Pairs")
    print("=" * 80)
    
    reward_model = RewardModelAPI(
        language="multilingual",
        api_type="huggingface",
        api_key=os.environ.get("HF_API_TOKEN")
    )
    
    qa_pairs = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is the largest planet?", "answer": "Jupiter is the largest planet in our solar system"},
        {"question": "How do photosynthesis work?", "answer": "Photosynthesis is the process"},
    ]
    
    print(f"\nScoring {len(qa_pairs)} Q&A pairs...\n")
    scores = reward_model.score_batch_sync(qa_pairs)
    
    for qa, score in zip(qa_pairs, scores):
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Score: {score:.3f}\n")


def example3_filtering_with_threshold():
    """Example 3: Filtering Q&A pairs based on reward threshold"""
    print("=" * 80)
    print("Example 3: Filtering Q&A Pairs with Threshold")
    print("=" * 80)
    
    evaluator = create_reward_evaluator(
        language="multilingual",
        reward_threshold=0.5,  # Keep only pairs with score >= 0.5
        api_type="huggingface",
        api_key=os.environ.get("HF_API_TOKEN")
    )
    
    qa_pairs = [
        {"question": "What is Paris?", "answer": "Paris is a city"},
        {"question": "What is AI?", "answer": "Artificial Intelligence is a field of computer science"},
        {"question": "How are you?", "answer": "I am"},  # Poor quality answer
    ]
    
    print(f"\nEvaluating {len(qa_pairs)} Q&A pairs with threshold 0.5...\n")
    filtered_pairs, scores = evaluator.evaluate_qa_pairs(qa_pairs)
    
    print(f"Original pairs: {len(qa_pairs)}")
    print(f"Filtered pairs: {len(filtered_pairs)}\n")
    
    for pair in filtered_pairs:
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer']}")
        print(f"Score: {pair['reward_score']:.3f}\n")


def example4_arabic_specific():
    """Example 4: Arabic-specific Q&A evaluation"""
    print("=" * 80)
    print("Example 4: Arabic-Specific Q&A Evaluation")
    print("=" * 80)
    
    evaluator = create_reward_evaluator(
        language="arabic",
        reward_threshold=0.55,
        api_type="huggingface",
        api_key=os.environ.get("HF_API_TOKEN")
    )
    
    qa_pairs = [
        {
            "question": "ما هي عاصمة فرنسا؟",
            "answer": "باريس هي عاصمة فرنسا وتقع على نهر السين"
        },
        {
            "question": "كم عدد أيام السنة؟",
            "answer": "365 يوم"
        },
        {
            "question": "ما هو الذهب؟",
            "answer": "معدن"  # Too short
        },
    ]
    
    print(f"\nEvaluating {len(qa_pairs)} Arabic Q&A pairs...\n")
    filtered_pairs, scores = evaluator.evaluate_qa_pairs(qa_pairs)
    
    print(f"Filtered to {len(filtered_pairs)}/{len(qa_pairs)} pairs\n")
    
    for pair in filtered_pairs:
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer']}")
        print(f"Score: {pair['reward_score']:.3f}\n")


def example5_using_local_vllm():
    """Example 5: Using local vLLM for reward scoring"""
    print("=" * 80)
    print("Example 5: Using Local vLLM for Reward Scoring")
    print("=" * 80)
    
    # Make sure vLLM server is running:
    # python -m vllm.entrypoints.openai.api_server --model OpenAssistant/reward-model-deberta-v3-large-v2 --port 8001
    
    try:
        reward_model = RewardModelAPI(
            language="multilingual",
            api_type="vllm",
            api_endpoint="http://localhost:8001/v1"
        )
        
        qa_pairs = [
            {"question": "What is Python?", "answer": "Python is a programming language"},
            {"question": "What is ML?", "answer": "Machine Learning is a subset of AI"},
        ]
        
        print("\nScoring with local vLLM...\n")
        scores = reward_model.score_batch_sync(qa_pairs)
        
        for qa, score in zip(qa_pairs, scores):
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer']}")
            print(f"Score: {score:.3f}\n")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure vLLM server is running:")
        print("python -m vllm.entrypoints.openai.api_server \\")
        print("  --model OpenAssistant/reward-model-deberta-v3-large-v2 \\")
        print("  --port 8001")


def example6_custom_reward_functions():
    """Example 6: Using custom reward functions"""
    print("=" * 80)
    print("Example 6: Using Custom Reward Functions")
    print("=" * 80)
    
    # Format answers as expected by reward functions
    completions = [
        [{"content": "This is a well-written, detailed answer about the topic."}],
        [{"content": "Short."}],
        [{"content": "This is another comprehensive answer that provides good information."}],
    ]
    
    print("\nScoring with different reward functions:\n")
    
    # Test length reward
    print("--- Length Reward (10-500 characters) ---")
    length_scores = answer_length_reward_func(completions, min_length=10, max_length=500)
    for comp, score in zip(completions, length_scores):
        print(f"Text: {comp[0]['content'][:40]}...")
        print(f"Score: {score:.3f}\n")
    
    # Test language quality
    print("\n--- Language Quality Reward ---")
    quality_scores = language_quality_reward_func(completions, language='english')
    for comp, score in zip(completions, quality_scores):
        print(f"Text: {comp[0]['content'][:40]}...")
        print(f"Score: {score:.3f}\n")
    
    # Test composite
    print("\n--- Composite Reward (Length + Quality) ---")
    composite_scores = composite_reward_func(
        completions,
        weights={'length': 0.5, 'language_quality': 0.5},
        language='english'
    )
    for comp, score in zip(completions, composite_scores):
        print(f"Text: {comp[0]['content'][:40]}...")
        print(f"Score: {score:.3f}\n")


def example7_processing_jsonl_file():
    """Example 7: Processing a JSONL file with reward scoring"""
    print("=" * 80)
    print("Example 7: Processing JSONL File with Reward Scoring")
    print("=" * 80)
    
    # Example: Read from synthetic data file and apply reward scoring
    input_file = "data/synthetic_data.jsonl"
    output_file = "data/synthetic_data_filtered.jsonl"
    
    if not os.path.exists(input_file):
        print(f"\nWarning: {input_file} not found. Creating example...")
        
        # Create sample data
        sample_data = [
            {
                "instruction": "What is AI?",
                "input": "",
                "output": "Artificial Intelligence is a field of computer science"
            },
            {
                "instruction": "What is ML?",
                "input": "",
                "output": "ML"  # Poor quality
            },
        ]
        
        os.makedirs(os.path.dirname(input_file) or ".", exist_ok=True)
        with open(input_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Process the file
    evaluator = create_reward_evaluator(
        language="english",
        reward_threshold=0.5,
        api_type="huggingface",
        api_key=os.environ.get("HF_API_TOKEN")
    )
    
    print(f"\nProcessing {input_file}...\n")
    
    filtered_count = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        with open(input_file, 'r', encoding='utf-8') as inp:
            for line_num, line in enumerate(inp, 1):
                try:
                    item = json.loads(line)
                    question = item.get('instruction', '')
                    answer = item.get('output', '')
                    
                    # Score the pair
                    score = evaluator.score_single_answer(question, answer)
                    
                    if score >= evaluator.reward_model.reward_threshold:
                        item['reward_score'] = score
                        out.write(json.dumps(item, ensure_ascii=False) + '\n')
                        filtered_count += 1
                    
                    print(f"Line {line_num}: Score {score:.3f} - {'✓ Keep' if score >= 0.5 else '✗ Skip'}")
                    
                except Exception as e:
                    print(f"Line {line_num}: Error - {e}")
    
    print(f"\nFiltered {filtered_count} items to {output_file}")


def example8_multilingual_processing():
    """Example 8: Processing multilingual data"""
    print("=" * 80)
    print("Example 8: Multilingual Processing")
    print("=" * 80)
    
    evaluator = create_reward_evaluator(
        language="multilingual",
        reward_threshold=0.5,
        api_type="huggingface",
        api_key=os.environ.get("HF_API_TOKEN")
    )
    
    qa_pairs = [
        {
            "question": "What is Python?",
            "answer": "Python is a high-level programming language known for readability"
        },
        {
            "question": "ما هو بايثون؟",
            "answer": "بايثون هي لغة برمجة عالية المستوى معروفة بسهولة قراءتها"
        },
        {
            "question": "¿Qué es Python?",
            "answer": "Python es un lenguaje de programación"
        },
    ]
    
    print("\nProcessing multilingual Q&A pairs...\n")
    filtered_pairs, scores = evaluator.evaluate_qa_pairs(qa_pairs)
    
    print(f"Processed {len(qa_pairs)} pairs, kept {len(filtered_pairs)}\n")
    
    for pair in filtered_pairs:
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer']}")
        print(f"Score: {pair['reward_score']:.3f}\n")


if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Basic API Scoring', example1_basic_api_scoring),
        '2': ('Batch Scoring', example2_batch_scoring),
        '3': ('Filtering with Threshold', example3_filtering_with_threshold),
        '4': ('Arabic-Specific', example4_arabic_specific),
        '5': ('Local vLLM', example5_using_local_vllm),
        '6': ('Custom Reward Functions', example6_custom_reward_functions),
        '7': ('JSONL File Processing', example7_processing_jsonl_file),
        '8': ('Multilingual Processing', example8_multilingual_processing),
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        example_num = sys.argv[1]
        print(f"\nRunning: Example {example_num} - {examples[example_num][0]}\n")
        examples[example_num][1]()
    else:
        print("\nReward Model Examples")
        print("=" * 80)
        print("\nUsage: python examples_reward_model.py <example_number>\n")
        print("Available examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print("\nExample:")
        print("  python examples_reward_model.py 1")
        print("\nNote: Some examples require HF_API_TOKEN environment variable:")
        print("  export HF_API_TOKEN='your_token_here'")
