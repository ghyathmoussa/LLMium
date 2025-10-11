"""
Battle AI Agent - Model Preparation and Fine-tuning
Optimized for AMD Instinct MI300X GPUs with Unsloth
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json
from rl_config import config

# ============================================================================
# STEP 1: MODEL SELECTION AND PREPARATION
# ============================================================================

class BattleAgentTrainer:
    def __init__(
        self,
        model_name=config["model"]["name"],
        max_seq_length=2048,
        load_in_4bit=True
    ):
        """
        Initialize the Battle Agent with Unsloth optimizations
        
        Args:
            model_name: Base model to use (Llama or Qwen recommended)
            max_seq_length: Maximum sequence length for training
            load_in_4bit: Use 4-bit quantization for memory efficiency
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load and prepare the model with Unsloth optimizations"""
        print(f"Loading model: {config['model']['name']}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config['model']['name'],
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect best dtype for MI300X
            load_in_4bit=self.load_in_4bit,
            # AMD MI300X specific optimizations
            device_map="auto",
        )
        
        # Apply PEFT (Parameter-Efficient Fine-Tuning) with QLoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=config['model']['hyperparameters']['r'],  # LoRA rank
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=config['model']['hyperparameters']['lora_alpha'],
            lora_dropout=config['model']['hyperparameters']['lora_dropout'],
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=config['project']['random_seed'],
            use_rslora=False,
            loftq_config=None,
        )
        
        print("Model loaded and PEFT configured successfully!")
        return self.model, self.tokenizer
    
    def prepare_qa_dataset(self, data_path=None, custom_data=None):
        """
        Prepare Q&A dataset for battle scenarios
        
        Args:
            data_path: Path to JSON/JSONL file with Q&A pairs
            custom_data: List of dicts with 'question' and 'answer' keys
        """
        if custom_data:
            data = custom_data
        elif data_path:
            with open(data_path, 'r') as f:
                if data_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
        else:
            # Example battle Q&A data
            data = [
                {
                    "question": "What is the capital of France?",
                    "answer": "The capital of France is Paris."
                },
                {
                    "question": "Explain quantum entanglement in simple terms.",
                    "answer": "Quantum entanglement is when two particles become connected so that the state of one instantly affects the other, no matter the distance between them."
                },
                {
                    "question": "Write a Python function to find prime numbers.",
                    "answer": "Here's a function:\n\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
                }
            ]
        
        # Format as instruction-following dataset
        formatted_data = []
        for item in data:
            text = f"""Below is a question that requires an answer. Provide a clear, accurate, and helpful response.

### Question:
{item['question']}

### Answer:
{item['answer']}"""
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        return dataset
    
    def train(
        self,
        dataset,
        output_dir="./battle_agent_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        max_steps=-1,
        logging_steps=10,
        save_steps=100
    ):
        """
        Fine-tune the model on Q&A data
        
        Args:
            dataset: Prepared dataset
            output_dir: Directory to save checkpoints
            Other args: Training hyperparameters
        """
        print("Starting training with Unsloth optimizations...")
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=logging_steps,
                optim="adamw_8bit",  # Memory-efficient optimizer
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                output_dir=output_dir,
                save_steps=save_steps,
                save_total_limit=2,
            ),
        )
        
        # Train the model
        trainer.train()
        
        print("Training completed!")
        return trainer
    
    def save_model(self, output_path="battle_agent_model"):
        """Save the fine-tuned model"""
        print(f"Saving model to {output_path}...")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved successfully!")
    
    def save_for_inference(self, output_path="battle_agent_inference"):
        """Save model in format optimized for inference"""
        print("Saving model for fast inference...")
        self.model.save_pretrained_merged(
            output_path,
            self.tokenizer,
            save_method="merged_16bit",  # or "merged_4bit" for smaller size
        )
        print(f"Inference model saved to {output_path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize trainer
    trainer = BattleAgentTrainer(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # or "unsloth/Qwen2.5-7B-bnb-4bit"
        max_seq_length=2048,
        load_in_4bit=True
    )
    
    # Load model with PEFT
    model, tokenizer = trainer.load_model()
    
    # Prepare your Q&A dataset
    # Option 1: Use custom data
    custom_qa_data = [
        {"question": "Your question here", "answer": "Your answer here"},
        # Add more Q&A pairs
    ]
    dataset = trainer.prepare_qa_dataset(custom_data=custom_qa_data)
    
    # Option 2: Load from file
    # dataset = trainer.prepare_qa_dataset(data_path="qa_data.json")
    
    # Train the model
    trainer.train(
        dataset=dataset,
        output_dir="./battle_agent_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
    )
    
    # Save the model
    trainer.save_model("battle_agent_final")
    trainer.save_for_inference("battle_agent_inference")
    
    print("\n✓ Battle Agent training complete!")
    print("Next steps: Run the inference script to test your agent")