"""
Battle AI Agent - Reinforcement Learning from Human Feedback (RLHF)
Advanced training with PPO for competition optimization
"""

import torch
from unsloth import FastLanguageModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from datasets import Dataset
import numpy as np
from rl_config import config
from reward_functions import create_reward_function

class BattleAgentRL:
    def __init__(
        self,
        model_path,
        max_seq_length=config['model']['hyperparameters']['max_seq_length'],
        load_in_4bit=config['model']['pretrained']['load_in_4bit']
    ):
        """
        Initialize Battle Agent for RL training
        
        Args:
            model_path: Path to base/fine-tuned model
            max_seq_length: Maximum sequence length
            load_in_4bit: Use 4-bit quantization
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        
    def load_model_for_rl(self):
        """Load model with value head for PPO training"""
        print("Loading model for Reinforcement Learning...")
        
        # Load base model
        base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
            device_map=config['hardware']['device'],
        )
        
        # Add value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model
        )
        
        # Create reference model (frozen copy for KL divergence)
        self.ref_model = create_reference_model(self.model)
        
        print("RL model loaded successfully!")
        return self.model, self.tokenizer
    
    
    def prepare_rl_dataset(self, qa_data):
        """
        Prepare dataset for RL training
        
        Args:
            qa_data: List of dicts with 'question' and 'answer' keys
        
        Returns:
            Dataset for RL training
        """
        formatted_data = []
        
        for item in qa_data:
            query = f"""Below is a question that requires an answer. Provide a clear, accurate, and helpful response.

### Question:
{item['question']}

### Answer:
""" 
            formatted_data.append({
                "query": query,
                "question": item['question'],
                "correct_answer": item['answer']
            })
        
        return Dataset.from_list(formatted_data)
    
    def train_with_ppo(
        self,
        dataset,
        reward_function,
        output_dir=config['project']['output_dir'],
        num_epochs=10,
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        mini_batch_size=config['training']['mini_batch_size']
    ):
        """
        Train agent using PPO (Proximal Policy Optimization)
        
        Args:
            dataset: RL dataset
            reward_function: Function to compute rewards
            output_dir: Output directory
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            mini_batch_size: Mini batch size for PPO updates
        """
        print("Starting Reinforcement Learning with PPO...")
        
        # Configure PPO
        ppo_config = PPOConfig(
            model_name=self.model_path,
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            ppo_epochs=4,
            seed=config['project']['random_seed'],
            log_with=None,
            tracker_project_name="battle_agent_rl",
        )
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=dataset,
        )
        
        # Training loop
        generation_kwargs = {
            "max_new_tokens": config['training']['max_new_tokens'],
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(ppo_trainer.dataloader):
                query_tensors = batch["input_ids"]
                
                # Generate responses
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs
                )
                
                # Decode responses
                batch_responses = self.tokenizer.batch_decode(
                    response_tensors,
                    skip_special_tokens=True
                )
                
                # Compute rewards
                rewards = []
                for i, response in enumerate(batch_responses):
                    question = batch["question"][i]
                    correct_answer = batch["correct_answer"][i]
                    
                    # Extract answer from response
                    if "### Answer:" in response:
                        answer = response.split("### Answer:")[-1].strip()
                    else:
                        answer = response.strip()
                    
                    # Compute reward
                    reward = reward_function(question, answer, correct_answer)
                    rewards.append(torch.tensor(reward))
                
                # Run PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                print(f"stats: {stats}")
                # Log progress
                if batch_idx % 10 == 0:
                    mean_reward = np.mean([r.item() for r in rewards])
                    print(f"  Batch {batch_idx}: Mean Reward = {mean_reward:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_dir = f"{output_dir}/checkpoint_epoch_{epoch+1}"
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                print(f"  Checkpoint saved to {checkpoint_dir}")
        
        print("\nRL training completed!")
        return ppo_trainer
    
    def save_rl_model(self, output_path="battle_agent_rl_final"):
        """Save the RL-trained model"""
        print(f"Saving RL model to {output_path}...")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("RL model saved successfully!")


# ============================================================================
# ADVANCED: Direct Preference Optimization (DPO)
# ============================================================================

from trl import DPOTrainer, DPOConfig

class BattleAgentDPO:
    """Alternative to PPO: Direct Preference Optimization"""
    
    def __init__(self, model_path, max_seq_length=2048):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load model for DPO training"""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map=config['hardware']['device'],
        )
        
        # Apply PEFT
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=config['model']['hyperparameters']['r'],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=config['model']['hyperparameters']['lora_alpha'],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        return self.model, self.tokenizer
    
    def prepare_preference_data(self, preference_data):
        """
        Prepare preference dataset for DPO
        
        Args:
            preference_data: List of dicts with:
                - 'question': The question
                - 'chosen': Preferred answer
                - 'rejected': Non-preferred answer
        
        Returns:
            Dataset formatted for DPO
        """
        formatted_data = []
        
        for item in preference_data:
            prompt = f"""Below is a question that requires an answer. Provide a clear, accurate, and helpful response.

### Question:
{item['question']}

### Answer:
"""
            formatted_data.append({
                "prompt": prompt,
                "chosen": item['chosen'],
                "rejected": item['rejected']
            })
        
        return Dataset.from_list(formatted_data)
    
    def train_with_dpo(
        self,
        dataset,
        output_dir="./battle_agent_dpo_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        beta=0.1
    ):
        """
        Train with Direct Preference Optimization
        
        Args:
            dataset: Preference dataset
            output_dir: Output directory
            num_train_epochs: Number of epochs
            per_device_train_batch_size: Batch size
            learning_rate: Learning rate
            beta: DPO temperature parameter
        """
        print("Starting DPO training...")
        
        dpo_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            beta=beta,
            gradient_accumulation_steps=4,
            logging_steps=10,
            save_steps=100,
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        )
        
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Will create automatically
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        print("DPO training completed!")
        return trainer


# ============================================================================
# USAGE EXAMPLE - REINFORCEMENT LEARNING
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Battle AI Agent - Reinforcement Learning Training")
    print("="*60)
    
    # ========== PPO TRAINING ==========
    print("\n[Option 1: PPO Training]")
    
    # Initialize RL trainer
    rl_trainer = BattleAgentRL(
        model_path=config['model']['pretrained']['model_path'],  # Use model from config
        max_seq_length=2048
    )
    
    # Load model for RL
    model, tokenizer = rl_trainer.load_model_for_rl()
    
    # Prepare training data
    qa_data = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "question": "What is 5 + 7?",
            "answer": "12"
        },
        {
            "question": "Who invented the telephone?",
            "answer": "Alexander Graham Bell"
        },
        # Add more Q&A pairs for better training
    ]
    
    rl_dataset = rl_trainer.prepare_rl_dataset(qa_data)
    
    # Create reward function
    reward_fn = create_reward_function(reward_type="hybrid")
    
    # Train with PPO
    ppo_trainer = rl_trainer.train_with_ppo(
        dataset=rl_dataset,
        reward_function=reward_fn,
        output_dir="./battle_agent_ppo",
        num_epochs=10,
        batch_size=8,
        learning_rate=1.41e-5
    )
    
    # Save RL model
    rl_trainer.save_rl_model("battle_agent_rl_final")
    
    # ========== DPO TRAINING (ALTERNATIVE) ==========
    print("\n[Option 2: DPO Training]")
    
    # Initialize DPO trainer
    dpo_trainer = BattleAgentDPO(
        model_path=config['model']['pretrained']['model_path'],  # Use model from config
        max_seq_length=2048
    )
    
    # Load model
    dpo_trainer.load_model()
    
    # Prepare preference data
    preference_data = [
        {
            "question": "Explain photosynthesis.",
            "chosen": "Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
            "rejected": "Photosynthesis is when plants make food."
        },
        {
            "question": "What is Python?",
            "chosen": "Python is a high-level, interpreted programming language known for its simplicity and versatility, widely used in web development, data science, and AI.",
            "rejected": "Python is a programming language."
        },
        # Add more preference pairs
    ]
    
    dpo_dataset = dpo_trainer.prepare_preference_data(preference_data)
    
    # Train with DPO
    dpo_trainer.train_with_dpo(
        dataset=dpo_dataset,
        output_dir="./battle_agent_dpo",
        num_train_epochs=3,
        learning_rate=5e-5
    )
    
    print("\n✓ Reinforcement Learning training complete!")
    print("Your battle agent is now optimized for competition!")