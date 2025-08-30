# Finetune the model class
# Created By Ghyath Moussa
# Email:gheathmousa@gmail.com

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from trl import GRPOTrainer, GRPOConfig
from utils.reward_functions import format_reward_func, accuracy_reward_func
from utils.logger import setup_app_logger
import argparse

# System prompt for reasoning tasks, can be customized
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
logger = setup_app_logger(__name__)
# Provide the path to the dataset and the model name
args = argparse.ArgumentParser()

args.add_argument("--ft-type", type=str, default="reasoning")
args.add_argument("--use-quantization", type=str, default="lora")
args.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
args.add_argument("--output-dir", type=str, default="./arabic-reasoning-model")
args.add_argument("--dataset-name", type=str, default=None)
args.add_argument("--batch-size", type=int, default=8)
args.add_argument("--gradient-accumulation_steps", type=int, default=2)
args.add_argument("--learning-rate", type=float, default=2e-4)
args.add_argument("--num_train-epochs", type=int, default=3)
args.add_argument("--max-steps", type=int, default=-1)
args.add_argument("--prompt", type=str, default="")
args.add_argument("--max-length", type=int, default=4096)
args.add_argument("--padding-side", type=str, default="right")
args.add_argument("--beta", type=float, default=0.04) 
args.add_argument("--num-generations", type=int, default=4)
args.add_argument("--max-completion-length", type=int, default=128)
args.add_argument("--token", type=str, default=None, help="Hugging Face token for gated models")
args.add_argument("--distributed", action="store_true", help="Enable distributed training")
args.add_argument("--world-size", type=int, default=1, help="Number of processes for distributed training")
args.add_argument("--rank", type=int, default=0, help="Rank of the current process")
args.add_argument("--local-rank", type=int, default=0, help="Local rank of the current process")
args.add_argument("--master-addr", type=str, default="localhost", help="Master node address for distributed training")
args.add_argument("--master-port", type=str, default="12355", help="Master node port for distributed training")

args = args.parse_args()

# Set your parameters
MODEL_NAME = args.model_name
OUTPUT_DIR = args.output_dir
DATASET_NAME = args.dataset_name
HF_TOKEN = args.token

# Training parameters
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
LEARNING_RATE = args.learning_rate
NUM_TRAIN_EPOCHS = args.num_train_epochs
MAX_STEPS = args.max_steps
FT_TYPE = args.ft_type
BETA = args.beta
NUM_GENERATIONS = args.num_generations
MAX_COMPLETION_LENGTH = args.max_completion_length

# Distributed training parameters
DISTRIBUTED = args.distributed
WORLD_SIZE = args.world_size
RANK = args.rank
LOCAL_RANK = args.local_rank
MASTER_ADDR = args.master_addr
MASTER_PORT = args.master_port


def setup_distributed_training(rank, world_size, master_addr, master_port):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)


def cleanup_distributed_training():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


# Define the dataset class
class ReasoningDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=2048): # Added data to constructor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data # Store data passed to constructor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def preprocess(self, text): # This might not be directly used by GRPOTrainer if data is pre-formatted
        return self.tokenizer.encode(text,
                                     add_special_tokens=True,
                                     max_length=self.max_length,
                                     truncation=False)


class ReasoningModel:
    def __init__(
            self,
            model_name: str,
            output_dir: str,
            dataset_name: str,
            batch_size: int,
            gradient_accumulation_steps: int,
            learning_rate: float,
            num_train_epochs: int,
            max_steps: int,
            use_quantization: str,
            max_length: int, 
            padding_side: str,
            ft_type: str, 
            beta: float,
            num_generations: int,
            max_completion_length: int,
            token: str,
            distributed: bool = False,
            world_size: int = 1,
            rank: int = 0,
            local_rank: int = 0,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.use_quantization = use_quantization
        self.max_length = max_length 
        self.padding_side = padding_side
        self.ft_type = ft_type
        self.beta = beta
        self.num_generations = num_generations
        self.max_completion_length = max_completion_length
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        
        # Set device based on distributed training configuration
        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.token = token

        if "llama" in self.model_name.lower() and not self.token:
            raise ValueError(
                "The model you're trying to use appears to be a Llama model, which requires "
                "a Hugging Face authentication token. Please provide your token using the "
                "--token argument when running the script."
            )

    def _load_from_json(self, dataset_name: str):
        try:
            with open(dataset_name, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Error: Dataset file {dataset_name} not found. Please create a dummy JSON file for testing.")
            return None
        except json.JSONDecodeError:
            logger.error(f"Error: Dataset file {dataset_name} is not a valid JSON.")
            return None
            
        return data
    
    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set
        tokenizer.padding_side = self.padding_side 
        return tokenizer

    def _get_grpo_config(self,): 
        grpo_args = GRPOConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            logging_steps=10, # Reduced for faster feedback
            save_steps=1000,    # Reduced for faster feedback
            save_total_limit=2,
            optim="adamw_8bit", 
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            fp16=True, 
            remove_unused_columns=False, 
            gradient_checkpointing=True, 
            max_prompt_length=self.max_length, 
            max_completion_length=self.max_completion_length,
            num_generations=self.num_generations,
            beta=self.beta,
            loss_type="bnpo",
            log_completions=True, # Enable for debugging
            num_completions_to_print=2, # Print 2 completions per log step
            # Distributed training configurations
            local_rank=self.local_rank if self.distributed else -1,
            ddp_find_unused_parameters=False if self.distributed else None,
        )
        return grpo_args
    
    def load_dataset(self, dataset_name: str):
        raw_dataset = self._load_from_json(dataset_name) 

        processed_dataset = []
        if not isinstance(raw_dataset, list):
            logger.warning(f"Warning: Expected a list from _load_from_json, but got {type(raw_dataset)}. Check your JSON structure.")
            return [{"prompt": [{"role":"system", "content":SYSTEM_PROMPT}, {"role":"user", "content":"Error loading data."}], "solution": "Error"}]

        for item in raw_dataset:
            if not isinstance(item, dict):
                logger.warning(f"Warning: Skipping item, expected dict but got {type(item)}: {item}")
                continue
            
            if "problem" not in item or "solution" not in item: # Ensure your JSON has these keys
                logger.warning(f"Warning: Skipping item due to missing 'problem' or 'solution': {item}")
                continue

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["problem"]},
            ]
            processed_dataset.append({"prompt": prompt_messages, "solution": item["solution"]})
            
        if not processed_dataset: 
             return [{"prompt": [{"role":"system", "content":SYSTEM_PROMPT}, {"role":"user", "content":"Empty dataset."}], "solution": ""}]
        return processed_dataset

    def load_model(self):
        bnb_config = None
        if self.use_quantization == "lora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        

        # Configure device mapping for distributed training
        if self.distributed:
            # For distributed training, use device_map="auto" or specific GPU assignment
            device_map = {"": self.local_rank}
        else:
            device_map = self.device
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            token=self.token
        )

        # List of fallback models that work in Kaggle without tokens
        fallback_models = [
            "microsoft/DialoGPT-medium",
            "distilgpt2", 
            "EleutherAI/gpt-neo-1.3B",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ]
        
        model = None
        model_to_use = self.model_name
        
        # Try original model first, then fallbacks
        models_to_try = [self.model_name] + [m for m in fallback_models if m != self.model_name]
        
        for attempt_model in models_to_try:
            try:
                logger.info(f"Attempting to load model: {attempt_model}")
                model = AutoModelForCausalLM.from_pretrained(
                    attempt_model,
                    quantization_config=bnb_config,
                    device_map=self.device,
                    trust_remote_code=True,
                    token=self.token if attempt_model == self.model_name else None
                )
                model_to_use = attempt_model
                logger.info(f"Successfully loaded model: {model_to_use}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {attempt_model}: {e}")
                if "mamba_ssm" in str(e):
                    logger.info("Model requires mamba_ssm which has CUDA compatibility issues in Kaggle")
                elif "401" in str(e) or "Unauthorized" in str(e):
                    logger.info("Model requires authentication token")
                continue
        
        if model is None:
            raise RuntimeError("Could not load any model. Please check your internet connection and try again.")


        # Apply LoRA configuration for PEFT
        # Check if model is already a PeftModel, common if loading fine-tuned model with adapter
        if not isinstance(model, PeftModel):
            # Different models have different attention module names
            target_modules = self._get_target_modules(model_to_use)
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=target_modules, 
                lora_dropout=0.05, # Slightly increased dropout
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # Added use_gradient_checkpointing
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model
    
    def _get_target_modules(self, model_name):
        """Get appropriate target modules for LoRA based on model architecture"""
        model_name_lower = model_name.lower()
        
        if "gpt" in model_name_lower or "dialogpt" in model_name_lower:
            return ["c_attn", "c_proj"]
        elif "llama" in model_name_lower or "tinyllama" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "qwen" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "phi" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "dense"]
        else:
            # Default fallback - try common attention module names
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def train(self, ):
        model = self.load_model()
        tokenizer = self._load_tokenizer() 
        
        if self.ft_type == "grpo_reasoning":
            grpo_config = self._get_grpo_config()
            train_data = self.load_dataset(self.dataset_name)

            if not train_data:
                logger.error("Error: Training data is empty. Aborting training.")
                return

            reward_fns = [format_reward_func, accuracy_reward_func]
            
            trainer = GRPOTrainer(
                model=model, 
                args=grpo_config,
                train_dataset=train_data,
                reward_funcs=reward_fns,
                processing_class=tokenizer, 
                peft_config=model.peft_config["default"] if hasattr(model, "peft_config") and "default" in model.peft_config else None
            )
        
        trainer.train()
        trainer.save_model(self.output_dir)


class EvaluateModel:
    def __init__(self, model_name: str, output_dir: str = None, token: str = None): # Added output_dir for PEFT model
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token = token
        # self.padding_side = "right" # Defined in tokenizer loading

        if "llama" in self.model_name.lower() and not self.token:
            raise ValueError(
                "The model you're trying to use appears to be a Llama model, which requires "
                "a Hugging Face authentication token. Please provide your token using the "
                "--token argument when running the script."
            )

    def _load_tokenizer(self):
        # If evaluating a PEFT model, tokenizer should be from base model
        # If output_dir is provided, assume it's a PEFT model, load base tokenizer
        # Otherwise, load from model_name (could be a fully merged model)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token) # Try base model first
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir, token=self.token) # Fallback to output_dir if base fails

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # For generation, left padding is usually preferred
        return tokenizer
    
    def evaluate(self,
                 test_prompt_text: str, # Made prompt an argument
                 temperature: float = 0.3,
                 max_new_tokens: int = 512,
                 top_p: float = 0.9,
                 do_sample: bool = True
            ):
        
        if self.output_dir: # PEFT model evaluation
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, # Base model name
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16, # Ensure consistent dtype
                token=self.token,
            )
            model_to_eval = PeftModel.from_pretrained(base_model, self.output_dir)
            model_to_eval.eval() # Set to evaluation mode
        else: # Evaluating a fully merged model or base model without adapter
            model_to_eval = AutoModelForCausalLM.from_pretrained(
                self.model_name, # This could be output_dir if model was saved fully
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=self.token,
            )
            model_to_eval.eval()

        tokenizer = self._load_tokenizer()
        
        # Example test prompt (Arabic), user should pass their own
        # test_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        # ### Instruction:
        # قم بحل المعادلة التالية خطوة بخطوة: 3x² + 7x - 10 = 0

        # ### Response:
        # """
        # Construct prompt in conversational format if model expects it
        # For GRPO fine-tuned models, it likely expects the system prompt + user prompt
        eval_prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_prompt_text}
        ]
        
        # Apply chat template if available and appropriate for the model
        try:
            final_prompt_str = tokenizer.apply_chat_template(eval_prompt_messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logger.error(f"Could not apply chat template: {e}. Using simple concatenation.")
            final_prompt_str = SYSTEM_PROMPT + "\nUser: " + test_prompt_text + "\nAssistant:"


        inputs = tokenizer(final_prompt_str, return_tensors="pt").to(self.device)
        
        logger.info(f"Generating response for: {final_prompt_str}")

        with torch.no_grad(): # Ensure no gradients are computed during generation
            outputs = model_to_eval.generate(
                **inputs, # Pass all inputs from tokenizer
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id # Important for open-ended generation
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("--- Model Output ---")
        logger.info(decoded_output)
        logger.info("--------------------")
        # To get only the generated part:
        # generated_part = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # print("--- Generated Part Only ---")
        # print(generated_part)
        # print("-------------------------")


def main():
    """Main training function that can be called by distributed processes"""
    # Setup distributed training if enabled
    if DISTRIBUTED:
        setup_distributed_training(RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
    
    try:
        if not DISTRIBUTED or RANK == 0:  # Only log on main process
            logger.info(f"Starting fine-tuning with type: {FT_TYPE}")
            logger.info(f"Model: {MODEL_NAME}, Dataset: {DATASET_NAME}, Output: {OUTPUT_DIR}")
            if DISTRIBUTED:
                logger.info(f"Distributed training enabled with {WORLD_SIZE} processes")

        reasoning_model = ReasoningModel(
            model_name=MODEL_NAME,
            output_dir=OUTPUT_DIR,
            dataset_name=DATASET_NAME,
            batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            max_steps=MAX_STEPS,
            use_quantization=args.use_quantization,
            max_length=args.max_length,
            padding_side=args.padding_side,
            ft_type=FT_TYPE, 
            beta=BETA, 
            num_generations=NUM_GENERATIONS, 
            max_completion_length=MAX_COMPLETION_LENGTH,
            token=HF_TOKEN,
            distributed=DISTRIBUTED,
            world_size=WORLD_SIZE,
            rank=RANK,
            local_rank=LOCAL_RANK,
        )

        reasoning_model.train()
        
        if not DISTRIBUTED or RANK == 0:  # Only log and evaluate on main process
            logger.info(f"Training finished. Model saved to {OUTPUT_DIR}")

            # Example of evaluation after training
            logger.info("\nStarting evaluation...")
            # Assuming the fine-tuned model is saved in OUTPUT_DIR and has PEFT adapters
            # If you fully merged and saved, model_name for EvaluateModel would be OUTPUT_DIR and output_dir=None
            eval_model = EvaluateModel(model_name=MODEL_NAME, output_dir=OUTPUT_DIR, token=HF_TOKEN) 
            
            test_math_prompt = "Solve for x: 2x + 5 = 11"
            logger.info(f"Evaluating with prompt: {test_math_prompt}")
            eval_model.evaluate(test_prompt_text=test_math_prompt)

            test_reasoning_prompt = "If a train travels at 60 mph for 2 hours, and then at 40 mph for 1 hour, what is the total distance traveled?"
            logger.info(f"Evaluating with prompt: {test_reasoning_prompt}")
            eval_model.evaluate(test_prompt_text=test_reasoning_prompt)
    
    finally:
        # Clean up distributed training
        if DISTRIBUTED:
            cleanup_distributed_training()


if __name__ == "__main__":
    main()