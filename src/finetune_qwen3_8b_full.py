#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B model with extended vocabulary for semantic IDs.
Stage 2: Full fine-tuning - trains all model parameters.
"""

# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied.
from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip

import glob
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

import wandb
from src.device_manager import DeviceManager
from src.logger import setup_logger
from src.test_prompts import SYSTEM_PROMPT, TEST_PROMPTS

logger = setup_logger("finetune-qwen3-full", log_to_file=True)


@dataclass
class FullFineTuneConfig:
    """Configuration for Stage 2: Full fine-tuning."""

    # Model settings - Load from vocab extension checkpoint
    model_name: str = "models/qwen3_8b_vocab_extended/final"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None  # None for auto detection
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    random_state: int = 1368
    num_proc: int = 32
    enable_thinking: bool = False

    # System prompt for semantic IDs
    system_prompt: str = SYSTEM_PROMPT

    # Data settings
    category: str = "Video_Games"
    data_dir: Path = Path("data")
    use_full_dataset: bool = True  # Use entire dataset for Stage 2
    max_training_samples: Optional[int] = None  # None means use all data

    # Full finetuning parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    gradient_clip_norm: float = 1.0  # Maximum gradient norm for clipping
    max_steps: int = -1  # -1 means use num_train_epochs
    num_train_epochs: int = 3  # Train for 3 epochs
    warmup_ratio: float = 0.03  # 3% of total steps for warmup
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = False  # Disable to use more memory but slightly faster
    optim: str = "adamw_8bit"  # "adamw_8bit", "paged_adamw_8bit", "adamw_torch_fused"

    # Output settings
    output_dir: Path = Path("models/qwen3_8b_full_finetuned")
    logging_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    eval_samples: int = 10000
    save_strategy: str = "steps"
    save_steps: int = 5000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True  # So we automatically save the best checkpoint based on metric for best model
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Checkpoint resume settings
    resume_from_checkpoint: bool = False  # Auto-resume if checkpoint exists
    checkpoint_dir_pattern: str = "checkpoint-*"  # Pattern for finding checkpoints

    # Computed paths (set in __post_init__)
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None
    rec_val_path: Optional[Path] = None  # Recommendation validation data

    def __post_init__(self):
        """Post-initialization setup and validation."""
        if self.dtype is None:
            # Force bfloat16 on H100 for better performance
            self.dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set and validate data paths
        self.train_path = self.data_dir / "output" / f"{self.category}_conversations_train.parquet"
        self.val_path = self.data_dir / "output" / f"{self.category}_conversations_val.parquet"
        self.rec_val_path = self.data_dir / "output" / f"{self.category}_recommendation_val.parquet"

        # Validate that training data exists
        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.train_path}. "
                f"Please ensure you have run the data preparation scripts for category '{self.category}'."
            )

        # Validation data is optional, just log if missing
        if not self.val_path.exists():
            logger.warning(f"Validation data not found at {self.val_path}. Training without validation set.")
            self.eval_strategy = "no"
            self.load_best_model_at_end = False

        # Verify checkpoint exists
        stage1_checkpoint = Path(self.model_name)
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found at {stage1_checkpoint}. "
                f"Please run Stage 1 (finetune_qwen3_8b_vocab.py) first to initialize embeddings."
            )

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== Qwen3-8B Full Fine-tuning Configuration ===")
        logger.info("Stage 2: Full Model Training")

        # Model settings
        logger.info("Model Settings:")
        logger.info(f"  model_name: {self.model_name} (Stage 1 checkpoint)")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  load_in_4bit: {self.load_in_4bit}")
        logger.info(f"  gradient_checkpointing: {self.gradient_checkpointing}")
        logger.info(f"  random_state: {self.random_state}")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  train_path: {self.train_path}")
        logger.info(f"  val_path: {self.val_path}")
        logger.info(f"  rec_val_path: {self.rec_val_path}")
        logger.info(f"  use_full_dataset: {self.use_full_dataset}")
        logger.info(f"  max_training_samples: {self.max_training_samples or 'All'}")

        # Training parameters
        logger.info("Training Parameters (Stage 2):")
        logger.info(f"  learning_rate: {self.learning_rate} (low for full fine-tuning)")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  num_train_epochs: {self.num_train_epochs}")
        logger.info(f"  warmup_ratio: {self.warmup_ratio}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  lr_scheduler_type: {self.lr_scheduler_type}")
        logger.info(f"  optim: {self.optim}")

        # Output settings
        logger.info("Output Settings:")
        logger.info(f"  output_dir: {self.output_dir}")
        logger.info(f"  logging_steps: {self.logging_steps}")
        logger.info(f"  eval_strategy: {self.eval_strategy}")
        logger.info(f"  eval_steps: {self.eval_steps}")
        logger.info(f"  save_strategy: {self.save_strategy}")
        logger.info(f"  save_steps: {self.save_steps}")
        logger.info(f"  save_total_limit: {self.save_total_limit}")
        logger.info(f"  load_best_model_at_end: {self.load_best_model_at_end}")
        logger.info(f"  resume_from_checkpoint: {self.resume_from_checkpoint}")
        logger.info("============================================")


def get_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """
    Find the latest checkpoint in the output directory.

    Args:
        output_dir: Directory to search for checkpoints

    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    checkpoint_pattern = str(output_dir / "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)

    if not checkpoint_dirs:
        logger.info("No existing checkpoints found")
        return None

    # Sort by step number (extract number from checkpoint-XXXXX format)
    try:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(os.path.basename(x).split("-")[-1]))
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing checkpoint directories: {e}")
        return None


def load_vocab_extended_model(config: FullFineTuneConfig):
    """Load the model from Stage 1 checkpoint with extended vocabulary."""
    logger.info(f"Loading Stage 1 checkpoint from: {config.model_name}")

    # Load model and tokenizer from Stage 1
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )

    # Verify vocabulary was extended
    vocab_size = len(tokenizer)
    logger.info(f"Loaded model with vocabulary size: {vocab_size}")

    # Check for semantic ID tokens
    test_tokens = ["<|rec|>", "<|sid_start|>", "<|sid_end|>", "<|sid_0|>", "<|sid_1023|>"]
    for token in test_tokens:
        if token in tokenizer.get_vocab():
            logger.info(f"✓ Found token: {token}")
        else:
            logger.warning(f"⚠ Missing token: {token}")

    # Test tokenization of semantic IDs
    logger.info("Testing semantic ID tokenization")
    test_string = "<|rec|><|sid_start|><|sid_0|><|sid_256|><|sid_512|><|sid_768|><|sid_end|>"
    token_ids = tokenizer.encode(test_string, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    logger.info(f"Test string: {test_string}")
    logger.info(f"Token IDs: {token_ids}")
    logger.info(f"Tokens: {tokens}")

    # Verify semantic ID tokens are single tokens (not split)
    sid_tokens_found = [t for t in tokens if "sid_" in t or t in ["<|rec|>", "<|sid_start|>", "<|sid_end|>"]]
    logger.info(f"Semantic tokens found: {sid_tokens_found}")

    # Decode back to verify round-trip
    decoded_string = tokenizer.decode(token_ids, skip_special_tokens=False)
    if decoded_string == test_string:
        logger.info(f"Verified that encoded SID ({test_string}) matches decoded SID {decoded_string}")
    assert decoded_string == test_string, (
        f"Round-trip mismatch: encoded string: {test_string}, decoded string {decoded_string}"
    )

    # Note: Gradient checkpointing disabled due to Unsloth/Qwen3 compatibility issues
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for memory efficiency")
    else:
        model.gradient_checkpointing_disable()
        logger.info("Gradient checkpointing disabled")

    # Ensure all parameters are trainable for full fine-tuning
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        # Only set requires_grad for floating point tensors
        if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
            if not param.requires_grad:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                trainable_params += param.numel()
        else:
            # Skip non-floating point parameters (e.g., embeddings indices)
            logger.debug(f"Skipping non-float parameter: {name} with dtype {param.dtype}")

    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Verify model can generate SID tokens
    logger.info("=== Model Verification ===")
    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]
    vocab_size = len(tokenizer)

    logger.info(f"Model input embedding size: {input_size}")
    logger.info(f"Model output embedding size: {output_size}")
    logger.info(f"Tokenizer vocabulary size: {vocab_size}")

    if output_size != vocab_size:
        logger.error(f"❌ CRITICAL: Model output layer ({output_size}) != Tokenizer size ({vocab_size})")
        logger.error(f"Model CANNOT generate tokens beyond index {output_size - 1}")

        # Check if SID tokens are out of range
        sample_sid_id = tokenizer.encode("<|sid_100|>", add_special_tokens=False)[0]
        logger.error(f"Sample SID token ID: {sample_sid_id}")
        if sample_sid_id >= output_size:
            logger.error(f"SID tokens start at {sample_sid_id}, but model can only generate up to {output_size - 1}")
            logger.error("This explains why model generates SIDs character-by-character!")

        # Attempt to fix
        logger.info("Attempting to resize model to match tokenizer")
        model.resize_token_embeddings(vocab_size)
        new_output_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After resize - Output embedding size: {new_output_size}")

        if new_output_size == vocab_size:
            logger.info("✅ Model successfully resized - should be able to generate SID tokens now")
        else:
            logger.error("❌ Resize failed - model will not work correctly!")
    else:
        logger.info("✅ Model dimensions verified - can generate all vocabulary tokens")

    logger.info("===================================")

    return model, tokenizer


def load_sid_dataset(config: FullFineTuneConfig, tokenizer, split="train"):
    """
    Load the full conversation dataset with semantic IDs.

    Args:
        config: Configuration object
        tokenizer: Tokenizer to apply chat template
        split: "train" or "val" to load respective dataset
    """
    logger.info(f"Loading full semantic ID conversation dataset ({split})")

    if split == "train":
        data_path = config.train_path
    elif split == "val":
        data_path = config.val_path
    else:
        raise ValueError(f"Invalid split: {split}")

    if not data_path.exists():
        logger.warning(f"Dataset not found at {data_path}")
        return None

    logger.info(f"Loading from: {data_path}")

    dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(dataset):,} conversations")

    # Apply sampling if configured
    if not config.use_full_dataset and config.max_training_samples and split == "train":
        num_samples = min(len(dataset), config.max_training_samples)
        logger.info(f"Sampling {num_samples:,} examples for training")
        dataset = dataset.shuffle(seed=config.random_state).select(range(num_samples))
    elif split == "val":
        # Always limit validation set size for efficiency
        num_samples = min(len(dataset), config.eval_samples)
        logger.info(f"Using {num_samples:,} examples for validation")
        dataset = dataset.shuffle(seed=config.random_state).select(range(num_samples))

    # Apply chat template using map
    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=config.enable_thinking,
        )
        return {"text": text}

    logger.info(f"Applying chat template to conversations with {config.num_proc} processes")
    dataset = dataset.map(
        apply_chat_template,
        remove_columns=dataset.column_names,
        num_proc=config.num_proc,
        batch_size=1000,  # Process in larger batches for efficiency
        writer_batch_size=5000,  # Write larger batches to disk
    )

    logger.info(f"Prepared dataset with {len(dataset):,} examples")

    # Verify semantic IDs are present and show sample
    if len(dataset) > 0:
        sample_text = dataset[0]["text"]
        sid_count = sample_text.count("<|sid_start|>")
        if sid_count > 0:
            logger.info(f"✓ Verified: Semantic ID tokens found (sample has {sid_count} IDs)")
        else:
            logger.warning("⚠ Warning: No semantic ID tokens found in sample")

        # Log a sample of the chat template output
        logger.info("=" * 60)
        logger.info(f"Sample chat template output ({split}): {sample_text}")

        # Tokenize the sample to show token IDs
        sample_token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        logger.info(f"Sample token IDs: {sample_token_ids} (total token = {len(sample_token_ids)})")
        logger.info("=" * 60)

    return dataset


class DataInspectionCallback(TrainerCallback):
    """Inspect training data and tokenization at each logging step."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.trainer = None  # Will be set later
        self.first_batch_inspected = False

    def set_trainer(self, trainer):
        """Set the trainer after it's been created."""
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        """Inspect first batch at training start."""
        if not self.first_batch_inspected:
            self.first_batch_inspected = True
            logger.info("=" * 60)
            logger.info("=== Initial Training Data Inspection ===")
            logger.info("=" * 60)

            try:
                if self.trainer is None:
                    logger.info("Trainer not yet set, skipping initial inspection")
                    return

                train_dataloader = self.trainer.get_train_dataloader()

                # Get first batch
                for batch in train_dataloader:
                    logger.info(f"Batch keys: {batch.keys()}")
                    logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
                    logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")

                    # Inspect first example in batch
                    first_example = batch["input_ids"][0]
                    decoded = self.tokenizer.decode(first_example, skip_special_tokens=False)

                    # Show full token list
                    logger.info(f"Tokens (first example): {first_example.tolist()}")
                    logger.info(f"Decoded: {decoded}")

                    # Count SID tokens
                    token_list = first_example.tolist()
                    sid_tokens = sum(1 for t in token_list if 151672 <= t <= 152695)
                    logger.info(f"Number of SID tokens: {sid_tokens}")

                    break  # Just check first batch

            except Exception as e:
                logger.info(f"Could not inspect first batch: {e}")

            logger.info("=" * 60)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Inspect data at each logging step."""
        # Only log every logging_steps
        if state.global_step > 0 and state.global_step % args.logging_steps == 0:
            logger.info("=" * 60)
            logger.info(f"=== Data Inspection at Step {state.global_step} ===")
            logger.info("=" * 60)

            try:
                if self.trainer is None:
                    logger.info("Trainer not set, skipping inspection")
                    return

                train_dataloader = self.trainer.get_train_dataloader()

                # Get one batch from current position
                for i, batch in enumerate(train_dataloader):
                    if i == (state.global_step // args.gradient_accumulation_steps) % len(train_dataloader):
                        logger.info(f"Batch shape: {batch['input_ids'].shape}")

                        # Show first example
                        first_example = batch["input_ids"][0]
                        token_list = first_example.tolist()

                        # Count SID tokens
                        sid_tokens = sum(1 for t in token_list if 151672 <= t <= 152695)

                        logger.info(f"First example - SID tokens: {sid_tokens}, Total tokens: {len(token_list)}")
                        logger.info(f"Token IDs: {token_list}")

                        # Show decoded version (truncated for readability)
                        decoded = self.tokenizer.decode(first_example, skip_special_tokens=False)
                        logger.info(f"Decoded: {decoded}")

                        break

            except Exception as e:
                logger.info(f"Could not inspect batch at step {state.global_step}: {e}")

            logger.info("=" * 60)


class CheckpointCallback(TrainerCallback):
    """Log checkpoint saves with disk space info."""

    def __init__(self, config):
        self.config = config
        self.checkpoint_count = 0

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        self.checkpoint_count += 1
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        checkpoint_size_gb = 0
        if os.path.exists(checkpoint_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(checkpoint_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            checkpoint_size_gb = total_size / (1024**3)

        logger.info(f"Checkpoint saved: {checkpoint_path} ({checkpoint_size_gb:.2f} GB)")
        logger.info(f"Total checkpoints saved: {self.checkpoint_count}, Max kept: {self.config.save_total_limit}")


class ModelMonitorCallback(TrainerCallback):
    """Monitor training progress and log to W&B with clean formatting."""

    def __init__(self, config, monitor_interval=100):
        self.config = config
        self.monitor_interval = monitor_interval
        self.initial_loss = None
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_step = 0
        self.batch_start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        """Track batch start time."""
        self.batch_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics with clean formatting."""
        if logs and "loss" in logs:
            # Track initial loss for improvement calculation
            if self.initial_loss is None:
                self.initial_loss = logs["loss"]

            current_loss = logs["loss"]
            improvement = (self.initial_loss - current_loss) / self.initial_loss * 100 if self.initial_loss else 0

            lr = logs.get("learning_rate", 0)
            grad_norm = logs.get("grad_norm", 0)
            epoch = logs.get("epoch", 0)

            current_time = time.time()
            batch_time_ms = (current_time - self.batch_start_time) * 1000 if self.batch_start_time else 0
            time_elapsed = current_time - self.last_log_time
            steps_done = state.global_step - self.last_log_step

            effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
            total_samples_seen = state.global_step * effective_batch_size
            total_batches_processed = state.global_step * self.config.gradient_accumulation_steps

            samples_processed = steps_done * effective_batch_size
            samples_per_second = samples_processed / time_elapsed if time_elapsed > 0 else 0

            avg_step_time_ms = (time_elapsed * 1000) / steps_done if steps_done > 0 else 0

            self.last_log_time = current_time
            self.last_log_step = state.global_step

            log_str = (
                f"Step {state.global_step:05d} | Epoch {epoch:.2f} | lr: {lr:.2e} | loss: {current_loss:.4f} | "
                f"grad_norm: {grad_norm:.2f} | improvement: {improvement:+.1f}% | time: {batch_time_ms:.0f}ms/step | "
                f"samples/s: {samples_per_second:,.0f}"
            )

            logger.info(log_str)

            wandb_log = {
                "loss/train": current_loss,
                "metrics/learning_rate": lr,
                "metrics/gradient_norm": grad_norm,
                "metrics/improvement_pct": improvement,
                "metrics/batch_time_ms": batch_time_ms,
                "metrics/avg_step_time_ms": avg_step_time_ms,
                "metrics/samples_per_second": samples_per_second,
                "metrics/total_batches": total_batches_processed,
                "metrics/total_samples": total_samples_seen,
            }
            wandb.log(wandb_log, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics with clean formatting."""
        if metrics:
            eval_loss = metrics.get("eval_loss", 0)

            effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
            total_samples_seen = state.global_step * effective_batch_size
            total_batches_processed = state.global_step * self.config.gradient_accumulation_steps

            log_str = f"EVAL | Step {state.global_step:05d} | loss: {eval_loss:.4f}"
            logger.info(log_str)

            wandb_log = {
                "loss/eval": eval_loss,
                "metrics/total_batches": total_batches_processed,
                "metrics/total_samples": total_samples_seen,
            }
            wandb.log(wandb_log, step=state.global_step)


class ConversationGenerationCallback(TrainerCallback):
    """Periodically test conversation generation during training."""

    def __init__(self, tokenizer, config, test_interval=500):
        self.tokenizer = tokenizer
        self.config = config
        self.test_interval = test_interval

        self.test_cases = []
        logger.info(f"Pre-processing {len(TEST_PROMPTS)} conversation test cases")

        for test_case in TEST_PROMPTS:
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                test_case["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config.enable_thinking,
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            self.test_cases.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "prompt_length": len(prompt),
                    "description": test_case["description"],
                    "expected": test_case["expected"],
                    "full_conversation": test_case["messages"],  # Store full conversation
                    "user_content": test_case["messages"][-1]["content"],  # For logging
                }
            )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Test generation at specified intervals."""
        if state.global_step % self.test_interval == 0 and state.global_step > 0:
            self.test_generation(model, state.global_step)

    def test_generation(self, model, step):
        """Test semantic ID generation capability."""
        logger.info("=" * 120)
        logger.info(f"Testing semantic ID generation at step {step}")
        logger.info("=" * 120)

        training_mode = model.training
        model.eval()

        results = []

        # Process test cases iteratively to avoid padding issues
        for i, test_case in enumerate(self.test_cases, 1):
            try:
                inputs = {
                    "input_ids": test_case["input_ids"].to(model.device),
                    "attention_mask": test_case["attention_mask"].to(model.device),
                }

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        min_p=0.01,
                        top_p=0.8,
                        top_k=20,
                    )

                # Extract generated token IDs (exclude input prompt)
                input_length = len(test_case["input_ids"][0])
                generated_ids = output[0][input_length:].tolist()
                input_ids = test_case["input_ids"][0].tolist()

                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
                response = generated_text[test_case["prompt_length"] :]

                expected = test_case["expected"]
                contains_expected = any(exp_part in response for exp_part in expected.split() if len(exp_part) > 3)

                results.append(
                    {
                        "step": step,
                        "test": test_case["description"],
                        "prompt": test_case["user_content"],
                        "response": response,
                        "matches_expected": contains_expected,
                    }
                )

                logger.info(f"Test {i}: {test_case['description']}")

                # Log full conversation for multi-turn dialogues
                if len(test_case.get("full_conversation", [])) > 2:  # More than system + 1 user message
                    logger.info("  Full conversation:")
                    for msg in test_case["full_conversation"]:
                        role = msg["role"]
                        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        logger.info(f"    [{role}]: {content}")
                else:
                    logger.info(f"  Prompt: {test_case['user_content']}")

                logger.info(f"  Input token IDs: {input_ids}")
                logger.info(f"  Expected: {expected}")
                logger.info(f"  Response: {response}")
                logger.info(f"  Response tokens: {generated_ids}")
                logger.info("-" * 120)

            except Exception as e:
                logger.warning(f"Generation failed for test {i}: {e}")
                results.append(
                    {
                        "step": step,
                        "test": test_case["description"],
                        "prompt": test_case["user_content"],
                        "response": f"[Error: {e}]",
                        "matches_expected": False,
                    }
                )

        model.train(training_mode)
        logger.info("=" * 60)


class RecommendationEvalCallback(TrainerCallback):
    """Evaluate semantic ID recommendation accuracy with hierarchical metrics."""

    def __init__(self, tokenizer, config, test_interval=500, eval_samples=500):
        self.tokenizer = tokenizer
        self.config = config
        self.test_interval = test_interval
        self.eval_samples = eval_samples
        self.eval_data = []

        if config.rec_val_path and config.rec_val_path.exists():
            logger.info(f"Loading recommendation validation data from {config.rec_val_path}")
            rec_val_df = pl.read_parquet(config.rec_val_path)
            logger.info(f"Loaded {len(rec_val_df):,} recommendation validation examples")

            if len(rec_val_df) > eval_samples:
                rec_val_df = rec_val_df.sample(n=eval_samples, seed=config.random_state)

            logger.info(f"Pre-processing {len(rec_val_df)} evaluation examples")

            for row in rec_val_df.iter_rows(named=True):
                messages = [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": row["user_message"]},
                ]

                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=config.enable_thinking
                )

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

                self.eval_data.append(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "prompt_length": len(prompt),  # For extracting response later
                        "expected": row["assistant_message"],
                        "user_message": row["user_message"],  # Keep for logging
                    }
                )

            logger.info(f"Pre-processed {len(self.eval_data)} evaluation examples")
        else:
            logger.warning(f"Recommendation validation data not found at {config.rec_val_path}")

    @staticmethod
    def parse_semantic_id(text: str) -> Optional[List[str]]:
        """
        Extract semantic ID tokens from text.

        Args:
            text: Text containing semantic ID

        Returns:
            List of SID tokens (e.g., ['48', '342', '687', '768']) or None if invalid
        """
        if not text:
            return None

        sid_pattern = r"<\|sid_start\|>((?:<\|sid_\d+\|>)+)<\|sid_end\|>"
        match = re.search(sid_pattern, text)

        if not match:
            return None

        tokens_str = match.group(1)
        token_pattern = r"<\|sid_(\d+)\|>"
        tokens = re.findall(token_pattern, tokens_str)

        # Valid semantic ID should have exactly 4 tokens
        if len(tokens) != 4:
            return None

        return tokens

    def log_sample_predictions(self, sample_predictions, max_samples=10):
        """
        Log a diverse set of sample predictions to console.

        Args:
            sample_predictions: List of prediction dictionaries
            max_samples: Maximum number of samples to log
        """
        if not sample_predictions:
            return

        logger.info("=" * 60)
        logger.info("Sample Predictions:")
        logger.info("=" * 60)

        perfect_matches = []
        partial_matches = []
        poor_matches = []
        invalid_format = []

        for sample in sample_predictions:
            results = sample["results"]
            if not results["valid_format"]:
                invalid_format.append(sample)
            elif results["level_1234_correct"]:
                perfect_matches.append(sample)
            elif results["level_12_correct"]:
                partial_matches.append(sample)
            else:
                poor_matches.append(sample)

        # Select diverse samples
        selected_samples = []

        # Add up to 3 perfect matches
        selected_samples.extend(perfect_matches[:3])

        # Add up to 3 partial matches
        selected_samples.extend(partial_matches[:3])

        # Add up to 2 poor matches
        selected_samples.extend(poor_matches[:2])

        # Add up to 2 invalid format examples
        selected_samples.extend(invalid_format[:2])

        # Limit to max_samples
        selected_samples = selected_samples[:max_samples]

        # Log each sample
        for i, sample in enumerate(selected_samples, 1):
            user_msg = sample["user_message"]
            results = sample["results"]
            accuracy_str = ""
            if results["valid_format"]:
                accuracy_str = f"L1:{'✓' if results['level_1_correct'] else '✗'} "
                accuracy_str += f"L2:{'✓' if results['level_12_correct'] else '✗'} "
                accuracy_str += f"L3:{'✓' if results['level_123_correct'] else '✗'} "
                accuracy_str += f"L4:{'✓' if results['level_1234_correct'] else '✗'}"

                if results["level_1234_correct"]:
                    accuracy_str += " (Perfect!)"
            else:
                accuracy_str = "Invalid format"

            logger.info(f"[{i}] User: {user_msg}")

            if "input_ids_last20" in sample:
                logger.info(f"    Input token IDs (last 20): {sample['input_ids_last20']}")
            logger.info(f"    Expected:  {sample['expected']}")
            logger.info(f"    Predicted: {sample['predicted']}")
            if "generated_ids" in sample:
                logger.info(f"    Predicted tokens: {sample['generated_ids']}")
            logger.info(f"    Accuracy:  {accuracy_str}")

        logger.info("=" * 60)

    @staticmethod
    def evaluate_prediction(predicted_text: str, expected_text: str) -> Dict[str, bool]:
        """
        Evaluate semantic ID prediction at multiple hierarchy levels.

        Args:
            predicted_text: Generated text containing predicted semantic ID
            expected_text: Ground truth text containing expected semantic ID

        Returns:
            Dictionary with evaluation results at each level
        """
        pred_tokens = RecommendationEvalCallback.parse_semantic_id(predicted_text)
        expected_tokens = RecommendationEvalCallback.parse_semantic_id(expected_text)

        results = {
            "valid_format": pred_tokens is not None,
            "level_1_correct": False,
            "level_12_correct": False,
            "level_123_correct": False,
            "level_1234_correct": False,
        }

        if pred_tokens and expected_tokens:
            # Level 1: First token
            results["level_1_correct"] = pred_tokens[0] == expected_tokens[0]

            # Level 1+2: First two tokens
            results["level_12_correct"] = pred_tokens[:2] == expected_tokens[:2]

            # Level 1+2+3: First three tokens
            results["level_123_correct"] = pred_tokens[:3] == expected_tokens[:3]

            # Level 1+2+3+4: All four tokens
            results["level_1234_correct"] = pred_tokens == expected_tokens

        return results

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Evaluate recommendation accuracy at specified intervals."""
        if not self.eval_data:
            return

        if state.global_step % self.test_interval == 0 and state.global_step > 0:
            self.evaluate_recommendations(model, state.global_step)

    def evaluate_recommendations(self, model, step, batch_size=16):
        """Evaluate recommendation performance on validation set."""
        logger.info("=" * 60)
        logger.info(f"Evaluating recommendation accuracy at step {step}")
        logger.info("=" * 60)

        # Switch to eval mode
        training_mode = model.training
        model.eval()

        metrics = {
            "valid_format": 0,
            "level_1_acc": 0,
            "level_12_acc": 0,
            "level_123_acc": 0,
            "level_1234_acc": 0,
            "total": 0,
        }

        sample_predictions = []

        # Process iteratively to avoid padding issues
        for idx, data in enumerate(self.eval_data):
            metrics["total"] += 1

            try:
                inputs = {
                    "input_ids": data["input_ids"].to(model.device),
                    "attention_mask": data["attention_mask"].to(model.device),
                }

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=8,  # Semantic IDs are short
                        temperature=0.7,
                        min_p=0.01,
                        top_p=0.8,
                        top_k=20,
                    )

                input_length = len(data["input_ids"][0])
                generated_ids = output[0][input_length:].tolist()

                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
                response = generated_text[data["prompt_length"] :]

                results = self.evaluate_prediction(response, data["expected"])
                for key, value in results.items():
                    if value and key != "valid_format":
                        metrics[key.replace("_correct", "_acc")] += 1
                    elif key == "valid_format" and value:
                        metrics["valid_format"] += 1

                if len(sample_predictions) < 50:  # Collect up to 50 samples
                    sample_predictions.append(
                        {
                            "user_message": data["user_message"],
                            "expected": data["expected"],
                            "predicted": response,
                            "generated_ids": generated_ids,  # Add token IDs
                            "input_ids_last20": data["input_ids"][0].tolist(),  # Last 20 input tokens
                            "results": results,
                        }
                    )

            except Exception as e:
                logger.warning(f"Generation failed for example {idx}: {e}")

        if metrics["total"] > 0:
            pct_metrics = {
                "valid_format_pct": metrics["valid_format"] / metrics["total"] * 100,
                "level_1_acc_pct": metrics["level_1_acc"] / metrics["total"] * 100,
                "level_12_acc_pct": metrics["level_12_acc"] / metrics["total"] * 100,
                "level_123_acc_pct": metrics["level_123_acc"] / metrics["total"] * 100,
                "level_1234_acc_pct": metrics["level_1234_acc"] / metrics["total"] * 100,
            }

            logger.info(f"Recommendation Evaluation Results (n={metrics['total']}):")
            logger.info(f"  Valid SID format: {pct_metrics['valid_format_pct']:.1f}%")
            logger.info(f"  Level 1 accuracy: {pct_metrics['level_1_acc_pct']:.1f}% (first token)")
            logger.info(f"  Level 1+2 accuracy: {pct_metrics['level_12_acc_pct']:.1f}% (first two tokens)")
            logger.info(f"  Level 1+2+3 accuracy: {pct_metrics['level_123_acc_pct']:.1f}% (first three tokens)")
            logger.info(f"  Level 1+2+3+4 accuracy: {pct_metrics['level_1234_acc_pct']:.1f}% (all four tokens)")

            self.log_sample_predictions(sample_predictions, max_samples=10)
            wandb_metrics = {
                "rec_eval/valid_format_pct": pct_metrics["valid_format_pct"],
                "rec_eval/level_1_acc": pct_metrics["level_1_acc_pct"],
                "rec_eval/level_12_acc": pct_metrics["level_12_acc_pct"],
                "rec_eval/level_123_acc": pct_metrics["level_123_acc_pct"],
                "rec_eval/level_1234_acc": pct_metrics["level_1234_acc_pct"],
                "rec_eval/n_samples": metrics["total"],
            }
            wandb.log(wandb_metrics, step=step)

        model.train(training_mode)
        logger.info("=" * 60)


def finetune_model(model, tokenizer, config: FullFineTuneConfig):
    """
    Perform full fine-tuning on the model with extended vocabulary.

    Args:
        model: The model with extended vocabulary from Stage 1
        tokenizer: The tokenizer with semantic ID tokens
        config: Training configuration
    """
    logger.info("Starting Stage 2: Full model fine-tuning")

    train_dataset = load_sid_dataset(config, tokenizer, split="train")
    val_dataset = load_sid_dataset(config, tokenizer, split="val")

    if not train_dataset:
        raise ValueError("Training dataset could not be loaded")

    total_steps = (
        len(train_dataset) * config.num_train_epochs // (config.batch_size * config.gradient_accumulation_steps)
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    logger.info(f"Total training steps: {total_steps:,}, Warmup steps: {warmup_steps:,}")

    sft_config = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=config.num_proc,  # Increase parallel tokenization processes
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.gradient_clip_norm,
        seed=config.random_state,
        output_dir=str(config.output_dir),
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
        completion_only_loss=False,
        log_level="error",  # Only log errors, not info/debug
        dataloader_num_workers=16,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster GPU transfer
        dataloader_prefetch_factor=8,  # Prefetch next batches
        # Additional TRL optimizations
        packing=False,  # Disabled as Unsloth: Hugging Face's packing is currently buggy - we're disabling it for now!
        max_length=config.max_seq_length,  # Explicit max length
        padding_free=False,  # Disabled as it does not work with Unsloth
        # Evaluation settings
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if val_dataset else None,
        per_device_eval_batch_size=config.batch_size,
        metric_for_best_model=config.metric_for_best_model if val_dataset else None,
        greater_is_better=config.greater_is_better,
        load_best_model_at_end=config.load_best_model_at_end and val_dataset is not None,
    )

    # Create data inspection callback (will set trainer later)
    data_inspection_callback = DataInspectionCallback(tokenizer)

    # Create other callbacks
    callbacks = [
        data_inspection_callback,  # Inspect data at each logging step
        CheckpointCallback(config),
        ModelMonitorCallback(config, monitor_interval=config.logging_steps),
        ConversationGenerationCallback(tokenizer, config, test_interval=config.eval_steps),
        RecommendationEvalCallback(tokenizer, config, test_interval=config.eval_steps, eval_samples=500),
    ]

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=callbacks,
    )

    # Now set the trainer in the data inspection callback
    data_inspection_callback.set_trainer(trainer)

    # Show current memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")

    resume_checkpoint = None
    if config.resume_from_checkpoint:
        resume_checkpoint = get_latest_checkpoint(config.output_dir)
        if resume_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            logger.info("Previous training progress will be continued")
        else:
            logger.info("Starting fresh training - no checkpoints found")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Show final memory and time stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        training_percentage = round(used_memory_for_training / max_memory * 100, 3)
        logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        logger.info(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_training} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {training_percentage} %.")

    wandb.summary["final_loss"] = trainer_stats.metrics.get("train_loss", None)
    wandb.summary["best_eval_loss"] = trainer_stats.metrics.get("eval_loss", None)
    wandb.summary["total_steps"] = trainer_stats.global_step
    wandb.summary["training_time_seconds"] = trainer_stats.metrics["train_runtime"]
    wandb.summary["training_time_minutes"] = trainer_stats.metrics["train_runtime"] / 60

    logger.info("Stage 2 full fine-tuning completed!")
    return trainer_stats


def save_final_model(model, tokenizer, config: FullFineTuneConfig):
    """Save the fully fine-tuned model."""
    logger.info("Saving fully fine-tuned model")

    final_save_path = config.output_dir / "final"
    final_save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to: {final_save_path}")

    model.save_pretrained(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))

    logger.info("Model and tokenizer saved successfully!")
    logger.info(f"Final checkpoint location: {final_save_path}")

    config_dict = {
        "stage": "full_finetuning",
        "base_checkpoint": config.model_name,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "category": config.category,
        "vocabulary_size": len(tokenizer),
        "use_full_dataset": config.use_full_dataset,
    }

    with open(final_save_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Training configuration saved")


if __name__ == "__main__":
    config = FullFineTuneConfig()
    device_manager = DeviceManager(logger)
    device = device_manager.device

    run_name = f"qwen3-8b-full-{config.category}-lr{config.learning_rate}"
    run = wandb.init(project="semantic-id-full-finetune", name=run_name, config=config.__dict__)
    config.log_config()

    model, tokenizer = load_vocab_extended_model(config)
    train_stats = finetune_model(model, tokenizer, config)
    save_final_model(model, tokenizer, config)

    wandb.finish()

    logger.info("=" * 50)
    logger.info("Stage 2: Full fine-tuning complete!")
    logger.info(f"Model saved to: {config.output_dir / 'final'}")
    logger.info("=" * 50)
