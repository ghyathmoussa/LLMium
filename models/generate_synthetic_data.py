import json
import os
import argparse
import openai # Added for Groq
import httpx # Import httpx for timeouts
import time # Added for rate limiting
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers.get_prompt import get_prompt
from utils.logger import setup_app_logger
from models.reward_model import RewardModelEvaluator, create_reward_evaluator
from utils.reward_functions import (
    answer_relevance_reward_func,
    answer_length_reward_func,
    language_quality_reward_func,
)
import re
import threading
from typing import Optional, List, Dict

logger = setup_app_logger(__name__)

def generate_qa_from_text_with_llm(text_content: str, num_qa_pairs: int = 3, api_key: str = None, llm_model: str = "llama3-8b-8192", api_url: str = None):
    """
    Generates question-answer pairs from the given Arabic text using an LLM via Groq or custom vLLM API.

    Args:
        text_content (str): The Arabic text from which to generate Q&A.
        num_qa_pairs (int): The desired number of Q&A pairs.
        api_key (str, optional): The API key. Defaults to None (not required for vLLM).
        llm_model (str, optional): The model to use. Defaults to "llama3-8b-8192".
        api_url (str, optional): Custom API URL (e.g., vLLM endpoint). Defaults to Groq.

    Returns:
        list: A list of dictionaries, where each dictionary is
              {"question": "...", "answer": "..."}.
              Returns an empty list if generation fails or no content.
    """
    if not text_content:
        return []

    # Determine the base URL
    if api_url:
        base_url = api_url
        # For custom vLLM, API key is optional - use a dummy if not provided
        if not api_key:
            api_key = "EMPTY"  # vLLM doesn't require authentication by default
            logger.info(f"Using custom API URL: {base_url} (without API key)")
    else:
        base_url = "https://api.groq.com/openai/v1"
        # For Groq, API key is required
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.error("Error: Groq API key not provided. Set GROQ_API_KEY environment variable or use the --llm-api-key argument.")
                return []

    try:
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(60.0, connect=10.0), # Added connect and read timeouts
        )
        prompt_text = get_prompt(language='arabic', num_qa_pairs=num_qa_pairs, text_content=text_content)

        logger.debug(f"DEBUG: Calling LLM with model {llm_model} for text starting with: {text_content[:100]}...")

        completion = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ]
        )
        
        response_content = completion.choices[0].message.content
        logger.debug(f"DEBUG: Raw LLM response: {response_content}") # For debugging
        logger.debug(f"length of prompt_text: {len(prompt_text)}")

        try:
            # Clean up the response to handle common LLM formatting issues like
            # markdown code blocks and trailing commas, which invalidate JSON.
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:].strip()
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3].strip()

            # Remove trailing commas before closing brackets/braces
            cleaned_response = re.sub(r',\s*([\]}])', r'\1', cleaned_response)
            
            qa_pairs = json.loads(cleaned_response)
            
            # Standardize keys from Arabic ("سؤال", "الإجابة") to English
            if qa_pairs and isinstance(qa_pairs, list):
                converted_pairs = []
                for pair in qa_pairs:
                    if isinstance(pair, dict):
                        question = pair.get("question") or pair.get("سؤال")
                        answer = pair.get("answer") or pair.get("الإجابة")
                        if question and answer:
                            converted_pairs.append({"question": question, "answer": answer})
                        else:
                            logger.warning(f"Skipping malformed Q&A pair after key conversion: {pair}")
                qa_pairs = converted_pairs

            if not isinstance(qa_pairs, list): # Ensure it's a list
                 logger.warning(f"Warning: LLM response was not a list of Q&A pairs. Response: {cleaned_response}")
                 return []
            return qa_pairs
        except json.JSONDecodeError:
            logger.error(f"Error: Could not decode LLM response as JSON even after cleaning. Cleaned Response: {cleaned_response}")
            # Attempt to extract Q&A pairs using a fallback mechanism if JSON parsing fails
            qa_list = []
            try:
                # This regex is improved to handle quasi-JSON responses.
                # It looks for quoted keys and captures the values.
                # Split by "question:" or "سؤال:" (case-insensitive) - improved to handle quotes and spaces
                potential_qa_blocks = re.split(r'["\']?(?:question|سؤال)["\']?\s*:', response_content, flags=re.IGNORECASE)
                
                for block in potential_qa_blocks:
                    if not block.strip():
                        continue

                    # Try to find "answer:" or "إجابة:" within the block - improved to handle quotes and spaces
                    match = re.search(r'(.*?)(?:["\']?(?:answer|إجابة)["\']?\s*:)\s*"(.*?)"', block, flags=re.IGNORECASE | re.DOTALL)
                    if match:
                        question_text = match.group(1).strip()
                        answer_text = match.group(2).strip()
                        
                        # Basic cleaning: remove potential leading/trailing quotes or list markers if model adds them
                        question_text = re.sub(r'^["\']?|["\']?,?$', '', question_text).strip()


                        if question_text and answer_text:
                            qa_list.append({"question": question_text, "answer": answer_text})
                        else:
                            logger.debug(f"Fallback: Found block but couldn't extract Q or A cleanly from: {block[:100]}...")
                    else:
                        logger.debug(f"Fallback: No clear answer found in block: {block[:100]}...")
                
                if qa_list:
                    logger.info(f"Successfully extracted {len(qa_list)} Q&A pairs using fallback regex from non-JSON response.")
                    return qa_list
                else:
                    logger.warning(f"Fallback regex extraction failed to find Q&A pairs in: {response_content}")
                    return []
            except Exception as e_fallback:
                logger.error(f"Error during fallback Q&A extraction: {e_fallback}")
                return []

    except openai.APIConnectionError as e:
        logger.error("Groq API connection error: The server could not be reached.")
        logger.error(f"Underlying error: {e.__cause__}")
        return []
    except openai.RateLimitError as e:
        logger.error(f"Groq API request exceeded rate limit. Raising error to stop process.")
        raise e # Re-raise to be handled by the caller
    except openai.APIStatusError as e:
        logger.error(f"Groq API returned an error status code: {e.status_code}")
        logger.error(f"Response: {e.response}")
        return []
    except openai.APIError as e:
        logger.error(f"Groq API error: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM call: {e}")
        return []


def process_single_chunk(
    data_entry,
    line_number,
    qa_per_chunk,
    api_key,
    llm_model,
    api_url,
    reward_evaluator: Optional[RewardModelEvaluator] = None,
):
    """
    Process a single chunk and generate Q&A pairs.
    Optionally evaluates pairs using a reward model.
    
    Returns a tuple of (success, qa_pairs, data_entry, line_number, error_message)
    """
    try:
        original_text = data_entry.get("text")
        if not original_text:
            return (False, [], data_entry, line_number, "Missing 'text' field")
        
        qa_pairs = generate_qa_from_text_with_llm(
            text_content=original_text,
            num_qa_pairs=qa_per_chunk,
            api_key=api_key,
            llm_model=llm_model,
            api_url=api_url
        )
        
        # Evaluate QA pairs using reward model if provided
        if reward_evaluator and qa_pairs:
            try:
                filtered_pairs, scores = reward_evaluator.evaluate_qa_pairs(qa_pairs)
                logger.debug(
                    f"Reward evaluation: {len(filtered_pairs)}/{len(qa_pairs)} "
                    f"pairs passed (scores: {[f'{s:.3f}' for s in scores]})"
                )
                qa_pairs = filtered_pairs
            except Exception as e:
                logger.warning(f"Error during reward evaluation: {e}. Continuing with unfiltered pairs.")
        
        return (True, qa_pairs, data_entry, line_number, None)
    except openai.RateLimitError as e:
        return (False, [], data_entry, line_number, f"Rate limit error: {e}")
    except Exception as e:
        return (False, [], data_entry, line_number, f"Error: {e}")


def create_synthetic_data(
    input_file_path,
    output_file_path,
    qa_per_chunk,
    api_key=None,
    llm_model=None,
    api_url=None,
    max_parallel_requests=1,
    use_reward_model: bool = False,
    reward_language: str = "multilingual",
    reward_threshold: float = 0.5,
    reward_api_type: str = "huggingface",
    reward_api_key: Optional[str] = None,
    reward_api_endpoint: Optional[str] = None,
):
    """
    Reads data from input_file_path, generates synthetic Q&A pairs,
    and writes them to output_file_path in an instruction-following format.
    The process is resumable and will skip already processed chunks.
    
    Args:
        input_file_path (str): Path to input JSONL file
        output_file_path (str): Path to output JSONL file
        qa_per_chunk (int): Number of Q&A pairs per chunk
        api_key (str, optional): API key for LLM service
        llm_model (str, optional): LLM model to use
        api_url (str, optional): Custom API URL for LLM
        max_parallel_requests (int): Number of parallel requests. Default is 1 (sequential).
        use_reward_model (bool): Whether to evaluate QA pairs with reward model. Default is False.
        reward_language (str): Language for reward model ('arabic', 'english', 'multilingual')
        reward_threshold (float): Minimum reward score to keep QA pair (0-1)
        reward_api_type (str): Type of API for reward model ('huggingface', 'openai', 'vllm', 'local')
        reward_api_key (str, optional): API key for reward model service
        reward_api_endpoint (str, optional): Custom endpoint for reward model
    """
    logger.info(f"Starting synthetic data generation from '{input_file_path}'...")
    logger.info(f"Max parallel requests: {max_parallel_requests}")
    
    # Initialize reward model if enabled
    reward_evaluator = None
    if use_reward_model:
        logger.info(f"Initializing reward model (language: {reward_language}, threshold: {reward_threshold})...")
        try:
            reward_evaluator = create_reward_evaluator(
                language=reward_language,
                reward_threshold=reward_threshold,
                api_type=reward_api_type,
                api_key=reward_api_key,
                api_endpoint=reward_api_endpoint,
            )
            logger.info("Reward model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize reward model: {e}. Continuing without reward model.")
            reward_evaluator = None
    
    processed_chunk_ids = set()
    if os.path.exists(output_file_path):
        logger.info(f"Output file '{output_file_path}' found. Resuming from last point.")
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if 'source_document_info' in entry and 'original_chunk_id' in entry['source_document_info']:
                            processed_chunk_ids.add(entry['source_document_info']['original_chunk_id'])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line in existing output file: {line.strip()}")
            logger.info(f"Loaded {len(processed_chunk_ids)} previously processed chunk IDs.")
        except Exception as e:
            logger.error(f"Could not read existing output file '{output_file_path}': {e}. Starting from scratch.")
            processed_chunk_ids = set()

    count_processed_lines = 0
    count_skipped_lines = 0
    count_generated_qa_pairs = 0
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Prepare chunks to process
    chunks_to_process = []
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, 1):
            try:
                data_entry = json.loads(line)
                chunk_id = data_entry.get("chunk_id")
                
                if chunk_id and chunk_id in processed_chunk_ids:
                    count_skipped_lines += 1
                    continue
                
                if not data_entry.get("text"):
                    logger.warning(f"Warning: Skipping line {line_number} due to missing 'text' field.")
                    continue
                
                chunks_to_process.append((data_entry, line_number))
            except json.JSONDecodeError:
                logger.warning(f"Warning: Skipping line {line_number} due to JSON decode error: {line.strip()}")
            except Exception as e:
                logger.error(f"Error reading line {line_number}: {line.strip()}. Error: {e}")
    
    logger.info(f"Found {len(chunks_to_process)} chunks to process (skipped {count_skipped_lines} already processed)")
    
    # Thread-safe lock for writing to file
    write_lock = threading.Lock()
    
    def write_results(outfile, qa_pairs, data_entry, line_number):
        """Write Q&A pairs to output file in a thread-safe manner"""
        nonlocal count_generated_qa_pairs, count_processed_lines
        
        original_text = data_entry.get("text", "")
        written_count = 0
        
        for qa in qa_pairs:
            if not (isinstance(qa, dict) and "question" in qa and "answer" in qa):
                logger.warning(f"Warning: Skipping malformed Q&A pair on line {line_number}: {qa}")
                continue
            
            instruction_data = {
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"],
                "source_document_info": {
                    "original_source": data_entry.get("source_document"),
                    "original_chunk_id": data_entry.get("chunk_id"),
                    "original_text_preview": original_text[:200] + "..."
                }
            }
            
            # Add reward score if available
            if "reward_score" in qa:
                instruction_data["reward_score"] = qa["reward_score"]
            
            with write_lock:
                outfile.write(json.dumps(instruction_data, ensure_ascii=False) + '\n')
                outfile.flush()  # Ensure data is written immediately
                count_generated_qa_pairs += 1
                written_count += 1
        
        with write_lock:
            count_processed_lines += 1
        
        return written_count
    
    # Process chunks in parallel or sequential
    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        if max_parallel_requests > 1:
            logger.info(f"Processing with {max_parallel_requests} parallel workers...")
            with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(
                        process_single_chunk,
                        data_entry,
                        line_number,
                        qa_per_chunk,
                        api_key,
                        llm_model,
                        api_url,
                        reward_evaluator,
                    ): (data_entry, line_number)
                    for data_entry, line_number in chunks_to_process
                }
                
                # Process completed tasks
                for future in as_completed(future_to_chunk):
                    data_entry, line_number = future_to_chunk[future]
                    try:
                        success, qa_pairs, data_entry, line_number, error_msg = future.result()
                        
                        if not success:
                            if "Rate limit" in str(error_msg):
                                logger.error(f"Rate limit error on line {line_number}. Consider reducing --max-parallel-requests")
                            else:
                                logger.warning(f"Failed to process line {line_number}: {error_msg}")
                            continue
                        
                        if not qa_pairs:
                            original_text = data_entry.get("text", "")
                            logger.warning(f"No Q&A pairs generated for line {line_number} (text starting: '{original_text[:50]}...')")
                            continue
                        
                        write_results(outfile, qa_pairs, data_entry, line_number)
                        
                        if count_processed_lines % 50 == 0:
                            logger.info(f"Processed {count_processed_lines} new lines. Generated {count_generated_qa_pairs} Q&A pairs...")
                    
                    except Exception as e:
                        logger.error(f"Error processing line {line_number}: {e}")
        else:
            logger.info("Processing sequentially...")
            for data_entry, line_number in chunks_to_process:
                success, qa_pairs, data_entry, line_number, error_msg = process_single_chunk(
                    data_entry, line_number, qa_per_chunk, api_key, llm_model, api_url, reward_evaluator
                )
                
                if not success:
                    if "Rate limit" in str(error_msg):
                        logger.error("API rate limit error received. Stopping generation. Progress has been saved.")
                        break
                    else:
                        logger.warning(f"Failed to process line {line_number}: {error_msg}")
                    continue
                
                if not qa_pairs:
                    original_text = data_entry.get("text", "")
                    logger.warning(f"No Q&A pairs generated for line {line_number} (text starting: '{original_text[:50]}...')")
                    continue
                
                write_results(outfile, qa_pairs, data_entry, line_number)
                
                if count_processed_lines % 50 == 0:
                    logger.info(f"Processed {count_processed_lines} new lines. Generated {count_generated_qa_pairs} Q&A pairs...")

    logger.info("Synthetic data generation complete or stopped.")
    logger.info(f"Processed {count_processed_lines} new lines from the input file.")
    logger.info(f"Skipped {count_skipped_lines} already processed lines.")
    logger.info(f"Generated a total of {count_generated_qa_pairs} new Q&A pairs this session.")
    logger.info(f"Output written to '{output_file_path}'")


if __name__ == "__main__":
    # --- Configuration ---
    # Default Groq model (example)
    DEFAULT_LLM_MODEL = "llama3-8b-8192" # You can change this
    QA_PAIRS_PER_CHUNK = 3
    DEFAULT_MAX_PARALLEL_REQUESTS = 1  # Default to sequential processing
    
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A data for fine-tuning with optional reward model filtering.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--qa-per-chunk", type=int, default=QA_PAIRS_PER_CHUNK, help="Number of Q&A pairs to generate per text chunk.")
    parser.add_argument("--llm-api-key", type=str, default=None, help="API key for the LLM service. If not provided, uses GROQ_API_KEY env var (not required for vLLM).")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help=f"The LLM model to use (default: {DEFAULT_LLM_MODEL}).")
    parser.add_argument("--api-url", type=str, default=None, help="Custom API URL (e.g., vLLM endpoint like http://localhost:8000/v1). If not provided, uses Groq API.")
    parser.add_argument("--max-parallel-requests", type=int, default=None, help=f"Number of parallel requests to send (default: 1 for sequential, can be set via MAX_PARALLEL_REQUESTS env var).")
    
    # Reward model arguments
    parser.add_argument("--use-reward-model", action="store_true", help="Enable reward model evaluation for Q&A pairs.")
    parser.add_argument("--reward-language", type=str, default="multilingual", choices=["arabic", "english", "multilingual"], help="Language for reward model (default: multilingual).")
    parser.add_argument("--reward-threshold", type=float, default=0.5, help="Minimum reward score to keep Q&A pair (0-1, default: 0.5).")
    parser.add_argument("--reward-api-type", type=str, default="huggingface", choices=["huggingface", "openai", "vllm", "local"], help="Type of API for reward model (default: huggingface).")
    parser.add_argument("--reward-api-key", type=str, default=None, help="API key for reward model service.")
    parser.add_argument("--reward-api-endpoint", type=str, default=None, help="Custom endpoint for reward model (e.g., for vLLM).")

    args = parser.parse_args()

    # Determine API key: use command-line arg if provided, otherwise it will be picked up from env var in the function.
    api_key_to_use = args.llm_api_key or os.environ.get("GROQ_API_KEY")
    
    # Determine max parallel requests: command-line > env var > default
    max_parallel = args.max_parallel_requests
    if max_parallel is None:
        max_parallel = int(os.environ.get("MAX_PARALLEL_REQUESTS", DEFAULT_MAX_PARALLEL_REQUESTS))
    
    # Determine reward API key
    reward_api_key = args.reward_api_key or os.environ.get("HF_API_TOKEN")
    
    create_synthetic_data(
        args.input_file, 
        args.output_file, 
        args.qa_per_chunk, 
        api_key=api_key_to_use,
        llm_model=args.llm_model,
        api_url=args.api_url,
        max_parallel_requests=max_parallel,
        use_reward_model=args.use_reward_model,
        reward_language=args.reward_language,
        reward_threshold=args.reward_threshold,
        reward_api_type=args.reward_api_type,
        reward_api_key=reward_api_key,
        reward_api_endpoint=args.reward_api_endpoint,
    )