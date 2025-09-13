# Unsloth Fine-Tuning Tutorial

Unsloth is the fastest and most memory-efficient fine-tuning framework, perfect for your GMKtec K11.

## 🚀 Why Unsloth?
- **2x faster** training than standard methods
- **70% less memory** usage
- **Easy to use** - just 3 lines of code to start
- **Great for AMD GPUs** with ROCm support

## 📋 Prerequisites
- Python 3.8+
- ROCm installed ✅ (you already have this)
- At least 8GB RAM (you have 32GB+ ✅)

## 🛠️ Installation

### Option 1: Conda (Recommended)
```bash
conda create -n unsloth python=3.11
conda activate unsloth
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Option 2: Pip
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## 📝 Your First Fine-Tuning Script

Create `first_finetune.py`:

```python
from unsloth import FastLanguageModel
import torch

# 1. Load a 4-bit quantized model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",  # Choose any from the Unsloth zoo
    max_seq_length=2048,  # Choose any - we auto support RoPE Scaling
    dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage
)

# 2. Add LoRA adapters for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0. Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None, # LoftQ
)

# 3. Prepare your dataset (example format)
dataset = [
    {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"instruction": "Explain photosynthesis", "output": "Photosynthesis is the process by which plants convert sunlight into energy..."},
    # Add more training examples
]

# 4. Format data for training
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

# 5. Train the model
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    formatting_func=formatting_prompts_func,
    args=TrainingArguments(
        per_device_train_batch_size=2,  # Adjust based on your GPU memory
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Number of training steps
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",  # Memory efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
    ),
)

# Start training
trainer_stats = trainer.train()
```

## 🎯 Supported Models

Unsloth supports many pre-quantized models:

```python
# Llama models
"unsloth/llama-2-7b-bnb-4bit"
"unsloth/llama-2-13b-bnb-4bit" 
"unsloth/codellama-7b-bnb-4bit"

# Mistral models  
"unsloth/mistral-7b-v0.1-bnb-4bit"
"unsloth/mistral-7b-instruct-v0.2-bnb-4bit"

# Gemma models
"unsloth/gemma-7b-bnb-4bit"
"unsloth/gemma-2b-bnb-4bit"

# Phi models (great for your hardware)
"unsloth/phi-3-mini-4k-instruct-bnb-4bit"
```

## 💾 Save Your Fine-tuned Model

```python
# Save locally
model.save_pretrained("my_finetuned_model")
tokenizer.save_pretrained("my_finetuned_model")

# Save to Hugging Face Hub (optional)
model.push_to_hub("your_username/model_name", token="your_hf_token")
```

## 🔧 Memory Optimization Tips for K11

```python
# For 32GB RAM systems (your setup)
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
max_seq_length = 2048

# For models larger than 7B, use these settings:
per_device_train_batch_size = 1  
gradient_accumulation_steps = 8
max_seq_length = 1024
```

## ⚡ ROCm Optimization

Since you have ROCm, add this to utilize AMD GPU:

```python
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"  # For RDNA3 architecture
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name()}")
```

## 🎯 Next Steps
1. Run the example script
2. Try different models
3. Experiment with your own datasets
4. Explore the `examples/` folder for more complex scenarios

## 📚 Useful Resources
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Model Zoo](https://huggingface.co/unsloth)
- [Discord Community](https://discord.gg/unsloth)