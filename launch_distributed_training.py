#!/usr/bin/env python3
"""
Distributed Training Launcher Script
Created By Ghyath Moussa
Email: gheathmousa@gmail.com

This script simplifies launching distributed training across multiple GPUs.
It automatically handles the process spawning and environment setup.
"""

import subprocess
import sys
import argparse
import torch
import os


def get_available_gpus():
    """Get the number of available GPUs"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def launch_distributed_training(
    script_path: str,
    num_gpus: int,
    master_addr: str = "localhost", 
    master_port: str = "12355",
    **kwargs
):
    """
    Launch distributed training using torch.distributed.launch
    
    Args:
        script_path: Path to the training script
        num_gpus: Number of GPUs to use
        master_addr: Master node address
        master_port: Master node port
        **kwargs: Additional arguments to pass to the training script
    """
    
    # Construct the base command
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={num_gpus}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script_path,
        "--distributed",
        f"--world-size={num_gpus}"
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    print(f"Launching distributed training with command:")
    print(" ".join(cmd))
    print(f"Using {num_gpus} GPUs")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(description="Launch distributed training")
    
    # Distributed training arguments
    parser.add_argument("--num-gpus", type=int, default=None, 
                       help="Number of GPUs to use (default: all available)")
    parser.add_argument("--master-addr", type=str, default="localhost",
                       help="Master node address")
    parser.add_argument("--master-port", type=str, default="12355", 
                       help="Master node port")
    
    # Training script arguments (these will be passed to finetune_model.py)
    parser.add_argument("--ft-type", type=str, default="grpo_reasoning",
                       help="Fine-tuning type")
    parser.add_argument("--use-quantization", type=str, default="lora",
                       help="Quantization method")
    parser.add_argument("--model-name", type=str, default="Qwen/QwQ-32B-Preview",
                       help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./arabic-reasoning-model",
                       help="Output directory for the trained model")
    parser.add_argument("--dataset-name", type=str, required=True,
                       help="Path to the training dataset")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per device (reduced for distributed training)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--num-train-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1,
                       help="Maximum training steps")
    parser.add_argument("--max-length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--padding-side", type=str, default="right",
                       help="Padding side for tokenizer")
    parser.add_argument("--beta", type=float, default=0.04,
                       help="Beta parameter for GRPO")
    parser.add_argument("--num-generations", type=int, default=4,
                       help="Number of generations for GRPO")
    parser.add_argument("--max-completion-length", type=int, default=128,
                       help="Maximum completion length")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face token for gated models")
    
    args = parser.parse_args()
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    if available_gpus == 0:
        print("Error: No GPUs available for distributed training")
        return 1
    
    # Determine number of GPUs to use
    num_gpus = args.num_gpus if args.num_gpus is not None else available_gpus
    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available")
        num_gpus = available_gpus
    
    if num_gpus == 1:
        print("Warning: Using only 1 GPU. Consider running without distributed training for single GPU.")
    
    print(f"Available GPUs: {available_gpus}")
    print(f"Using GPUs: {num_gpus}")
    
    # Path to the training script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, "models", "finetune_model.py")
    
    if not os.path.exists(training_script):
        print(f"Error: Training script not found at {training_script}")
        return 1
    
    # Prepare arguments for the training script
    training_args = {
        "ft-type": args.ft_type,
        "use-quantization": args.use_quantization,
        "model-name": args.model_name,
        "output-dir": args.output_dir,
        "dataset-name": args.dataset_name,
        "batch-size": args.batch_size,
        "gradient-accumulation-steps": args.gradient_accumulation_steps,
        "learning-rate": args.learning_rate,
        "num-train-epochs": args.num_train_epochs,
        "max-steps": args.max_steps,
        "max-length": args.max_length,
        "padding-side": args.padding_side,
        "beta": args.beta,
        "num-generations": args.num_generations,
        "max-completion-length": args.max_completion_length,
        "token": args.token,
    }
    
    # Launch distributed training
    return launch_distributed_training(
        script_path=training_script,
        num_gpus=num_gpus,
        master_addr=args.master_addr,
        master_port=args.master_port,
        **training_args
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
