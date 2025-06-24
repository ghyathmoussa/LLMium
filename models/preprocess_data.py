import json
import os
from transformers import AutoTokenizer
from utils.logger import setup_app_logger
import trankit
from tqdm import tqdm
import argparse

# --- Configuration ---
INPUT_FOLDER_PATH = "../source_data/"
OUTPUT_JSONL_PATH = "../data/processed_data.jsonl"
MAX_TOKENS_PER_CHUNK = 2048 # As per your requirement
FIRST_LINE_TO_SKIP = "هذا الملف آليا بواسطة المكتبة الشاملة"

logger = setup_app_logger(__name__)

class Processor:
    def __init__(self, model_name, max_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        # Initialize Trankit for Arabic
        self.nlp = trankit.Pipeline('arabic', cache_dir='../cache')

    def _split_text_to_fit(self, text, max_len):
        """
        Splits a given text into smaller pieces that are_ideally_ less than
        or equal to max_len tokens. Splits are attempted at sentence boundaries first,
        then word boundaries.
        A single word exceeding max_len will be returned as its own piece (and will be oversized).
        """
        # Try to split into sentences first using Trankit
        sentences = self.sentencize_text(text)
        
        output_pieces = []
        for sent in sentences:
            if not sent.strip(): # Skip empty sentences
                continue
            
            _, sent_tokens = self.tokenize_and_count(sent)
            if sent_tokens <= max_len:
                output_pieces.append(sent)
            else:
                # Sentence is too long, split by words
                logger.info(f"Sentence part is too long ({sent_tokens} tokens), splitting by words: '{sent[:100]}...'")
                words = sent.split() 
                if not words:
                    continue

                current_piece_text_for_word_split = ""
                for word_idx, word in enumerate(words):
                    text_to_try_adding = (" " + word) if current_piece_text_for_word_split else word
                    
                    prospective_new_piece = current_piece_text_for_word_split + text_to_try_adding
                    _, prospective_tokens = self.tokenize_and_count(prospective_new_piece)

                    if prospective_tokens <= max_len:
                        current_piece_text_for_word_split = prospective_new_piece
                    else:
                        if current_piece_text_for_word_split.strip():
                            output_pieces.append(current_piece_text_for_word_split)
                        
                        _, single_word_tokens = self.tokenize_and_count(word)
                        if single_word_tokens > max_len:
                            logger.warning(f"Word '{word[:50]}...' ({single_word_tokens} tokens) itself exceeds max_len ({max_len}). Adding as oversized piece.")
                            if word.strip(): output_pieces.append(word)
                            current_piece_text_for_word_split = "" 
                        else:
                            current_piece_text_for_word_split = word
                
                if current_piece_text_for_word_split.strip():
                    output_pieces.append(current_piece_text_for_word_split)
                    
        return [p for p in output_pieces if p.strip()]

    def tokenize_and_count(self, text_batch):
        """
        Tokenizer Function to return list of token counts for a batch of text
        """
        tokens = self.tokenizer.encode(text_batch)
        return tokens, len(tokens)

    # --- Helper Functions ---
    def clean_text(self, text):
        """Basic text cleaning."""
        text = text.strip()

        # remove the header line
        parts = text.split("\n\n", 1)
        if len(parts) > 1:
            text = parts[1]
        
        return text

    def sentencize_text(self, text):
        """
        Use Trankit to split text into sentences.
        """
        try:
            doc = self.nlp.ssplit(text)
            sentences = [sent['text'] for sent in doc['sentences']]
            return sentences
        except Exception as e:
            logger.warning(f"Error during sentence segmentation: {e}")
            # Fall back to simple splitting if Trankit fails
            return [text]

    def stream_semantic_units(self, file_path, line_to_skip=None):
        """
        Reads the input file and yields semantic units (sentences).
        Uses Trankit for sentence segmentation.
        Skips a specified first line if provided.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                logger.info(f"Reading file: {file_path}")
                text = f.read()
                text = self.clean_text(text)
                
                # First split by paragraphs to maintain some structure
                paragraphs = [p for p in text.split("\n") if p.strip()]
                print(len(paragraphs))
                for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
                    # Use Trankit to split paragraph into sentences
                    sentences = self.sentencize_text(paragraph)
                    for sentence in sentences:
                        if sentence.strip():
                            yield sentence.strip()
        except FileNotFoundError:
            logger.error(f"Error: Input file not found at {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading or processing file {file_path}: {e}")
            raise

    # --- Main Processing Logic ---
    def process_folder(self, input_folder_path, output_jsonl_path, skip_header_line=None):
        """
        Processes all .txt files in a given folder, chunks them, and writes to a single JSONL file.
        """
        all_chunks_data = []
        chunk_id_counter = 0

        logger.info(f"Searching for .txt files in '{input_folder_path}'...")
        try:
            txt_files = [f for f in os.listdir(input_folder_path) if f.endswith('.txt') and os.path.isfile(os.path.join(input_folder_path, f))]
        except FileNotFoundError:
            logger.error(f"Error: Input directory not found at {input_folder_path}")
            return
        
        logger.info(f"Found {len(txt_files)} .txt files to process.")

        for filename in tqdm(txt_files, desc="Processing files"):
            file_path = os.path.join(input_folder_path, filename)
            file_chunks, updated_chunk_id_counter = self._process_single_file(
                input_file_path=file_path,
                chunk_id_offset=chunk_id_counter,
                skip_header_line=skip_header_line
            )
            all_chunks_data.extend(file_chunks)
            chunk_id_counter = updated_chunk_id_counter

        # Write to JSONL file
        try:
            with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
                for entry in tqdm(all_chunks_data, desc="Writing chunks to JSONL"):
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write('\n')
            logger.info(f"Successfully processed and wrote {len(all_chunks_data)} chunks from {len(txt_files)} file(s) to {output_jsonl_path}")
        except IOError:
            logger.error(f"Error: Could not write to output file {output_jsonl_path}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during writing: {e}")

    def _process_single_file(self, input_file_path, chunk_id_offset=0, skip_header_line=None):
        """
        Processes a single input file, chunks it, and returns chunk data.
        """
        all_chunks_data = []
        current_chunk_text_parts = []
        current_chunk_token_count = 0
        chunk_id_counter = chunk_id_offset

        logger.info(f"Starting processing for {input_file_path}...")

        semantic_unit_generator = self.stream_semantic_units(input_file_path, skip_header_line)

        for i, sentence in enumerate(tqdm(semantic_unit_generator, desc=f"Processing sentences for {os.path.basename(input_file_path)}")):
            if not sentence:
                continue

            # New adaptive splitting logic
            processed_parts_for_sentence = []
            _, initial_token_count = self.tokenize_and_count(sentence)

            if initial_token_count > self.max_tokens:
                logger.warning(f"Semantic unit {i+1} (starts with '{sentence[:50]}...') has {initial_token_count} tokens, exceeding max_tokens ({self.max_tokens}). Attempting to split adaptively.")
                processed_parts_for_sentence = self._split_text_to_fit(sentence, self.max_tokens)
                if not processed_parts_for_sentence:
                    logger.warning(f"Splitting semantic unit '{sentence[:50]}...' resulted in no processable parts.")
                    continue
            else:
                processed_parts_for_sentence = [sentence]
            
            for sentence_part in processed_parts_for_sentence:
                if not sentence_part.strip():
                    continue

                _, sentence_token_count = self.tokenize_and_count(sentence_part)

                if sentence_token_count == 0 and sentence_part:
                    logger.warning(f"Warning: Tokenizer returned 0 tokens for non-empty sentence part: '{sentence_part[:100]}...'")
                    continue
                elif sentence_token_count == 0:
                    continue

                # Handle oversized single sentence parts
                # This might happen if _split_text_to_fit returns a single word that's too long.
                if sentence_token_count > self.max_tokens:
                    logger.warning(f"Warning: Sentence part (from unit {i+1}, original: '{sentence[:50]}...') has {sentence_token_count} tokens, exceeding max_tokens ({self.max_tokens}). Part: '{sentence_part[:50]}...'")
                    # The chunking logic below will handle this part, potentially as an oversized chunk.

                # If adding the current sentence_part would exceed the max token limit for the current chunk
                if current_chunk_token_count + sentence_token_count > self.max_tokens and current_chunk_text_parts:
                    # Finalize the current chunk
                    final_chunk_text = "\n".join(current_chunk_text_parts)
                    chunk_id_counter += 1
                    json_object = {
                        "text": final_chunk_text,
                        "source_document": input_file_path,
                        "chunk_id": f"chunk_{chunk_id_counter}",
                        "token_count_estimate": current_chunk_token_count
                    }
                    all_chunks_data.append(json_object)

                    # Start a new chunk with the current sentence_part
                    current_chunk_text_parts = [sentence_part]
                    current_chunk_token_count = sentence_token_count
                else:
                    # Add the current sentence_part to the current chunk
                    current_chunk_text_parts.append(sentence_part)
                    current_chunk_token_count += sentence_token_count

        # Add the last remaining chunk
        if current_chunk_text_parts:
            final_chunk_text = "\n".join(current_chunk_text_parts)
            chunk_id_counter += 1
            json_object = {
                "text": final_chunk_text,
                "source_document": input_file_path,
                "chunk_id": f"chunk_{chunk_id_counter}",
                "token_count_estimate": current_chunk_token_count
            }
            all_chunks_data.append(json_object)
        
        return all_chunks_data, chunk_id_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text data from a folder into a single JSONL file with adaptive chunking.")
    parser.add_argument('--input-dir', type=str, default=INPUT_FOLDER_PATH, help='Path to the folder containing input .txt files.')
    parser.add_argument('--output-file', type=str, default=OUTPUT_JSONL_PATH, help='Path to the output JSONL file.')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS_PER_CHUNK, help='Maximum tokens per chunk.')
    parser.add_argument('--skip-header', type=str, default=FIRST_LINE_TO_SKIP, help='First line to skip in the input file.')
    
    args = parser.parse_args()
    
    logger.info(f"Input folder: {args.input_dir}")
    logger.info(f"Output JSONL file: {args.output_file}")
    logger.info(f"Max tokens per chunk: {args.max_tokens}")
    logger.info(f"Skipping header line: {args.skip_header}")

    processor = Processor("aubmindlab/bert-base-arabertv02", args.max_tokens)
    
    processor.process_folder(args.input_dir, args.output_file, args.skip_header)
