# Step 3: Fine-tune Qwen2.5 0.6B

Time for the magic! Let's fine-tune your first model using Unsloth - optimized for speed and efficiency.

## 🎯 What Happens During Fine-tuning?

1. **Load Base Model**: Start with pre-trained Qwen2.5
2. **Add LoRA Adapters**: Small trainable layers (saves memory)  
3. **Feed Training Data**: Model learns your examples
4. **Monitor Progress**: Watch loss decrease over time
5. **Save Weights**: Keep your fine-tuned adapters

## 🚀 The Training Script

Create `train_qwen.py`:

```python
"""
Fine-tune Qwen2.5 0.6B for GenAI assistance
Optimized for GMKtec K11 - beginner friendly!
"""

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer

# AMD GPU optimization
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

class QwenFineTuner:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
        
    def load_model(self):
        """Load Qwen2.5 with Unsloth optimizations"""
        print("🚀 Loading Qwen2.5 0.5B with Unsloth...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,  # Reasonable for beginners
            dtype=None,  # Auto-detect best dtype
            load_in_4bit=True,  # Memory efficient
        )
        
        print("🔧 Adding LoRA adapters...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # Low rank for fast training
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # 0 is optimized
            bias="none",  # "none" is optimized
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        print("✅ Model loaded and configured!")
        self.model.print_trainable_parameters()
        
    def load_datasets(self):
        """Load training and validation datasets"""
        print("📊 Loading datasets...")
        
        # Load your created datasets
        with open("data/train_dataset.json", "r") as f:
            train_data = json.load(f)
        
        with open("data/val_dataset.json", "r") as f:
            val_data = json.load(f)
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        print(f"✅ Loaded {len(train_dataset)} training examples")
        print(f"✅ Loaded {len(val_dataset)} validation examples")
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset):
        """Run the fine-tuning process"""
        print("🏋️ Starting fine-tuning...")
        
        # Training configuration optimized for beginners and K11
        training_args = TrainingArguments(
            output_dir="./qwen_training_output",
            
            # Training schedule
            num_train_epochs=3,  # Good starting point
            per_device_train_batch_size=4,  # Balanced for 2GB VRAM
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,  # Effective batch size = 8
            
            # Learning rate
            learning_rate=2e-4,  # Good for LoRA
            warmup_steps=20,
            lr_scheduler_type="linear",
            
            # Monitoring
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            evaluation_strategy="steps",
            
            # Memory optimization
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            dataloader_num_workers=2,
            optim="adamw_8bit",  # Memory efficient
            
            # Model saving
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            report_to=None,  # Disable wandb for simplicity
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
        )
        
        # Start training!
        print("🎯 Training started! This will take 15-30 minutes...")
        print("💡 Watch the eval_loss - it should decrease over time")
        
        trainer_stats = trainer.train()
        
        print("✅ Training completed!")
        print(f"📊 Final training loss: {trainer_stats.training_loss:.4f}")
        
        return trainer
    
    def save_model(self):
        """Save the fine-tuned model"""
        print("💾 Saving fine-tuned model...")
        
        output_dir = "qwen_finetuned"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model saved to {output_dir}/")
        
    def test_model(self):
        """Quick test of the fine-tuned model"""
        print("🧪 Testing your fine-tuned model...")
        
        # Switch to inference mode
        FastLanguageModel.for_inference(self.model)
        
        test_questions = [
            "What is fine-tuning in machine learning?",
            "How do I choose the right model size?",
            "Best practices for GenAI projects?",
        ]
        
        for question in test_questions:
            # Format question using same template as training
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response:
"""
            
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            
            print(f"\n❓ Question: {question}")
            print("🤖 Answer: ", end="")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens (the answer)
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            print(response.strip())
            print("-" * 60)

def main():
    print("🎯 Fine-tuning Qwen2.5 for GenAI Assistance")
    print("🖥️  Hardware: GMKtec K11")
    print("⏱️  Expected time: 15-30 minutes")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU detected - training will be slow!")
    
    # Check data files
    if not Path("data/train_dataset.json").exists():
        print("❌ Training data not found!")
        print("   Run: python create_dataset.py first")
        return
    
    # Initialize trainer
    trainer = QwenFineTuner()
    
    try:
        # Step 1: Load model
        trainer.load_model()
        
        # Step 2: Load datasets  
        train_dataset, val_dataset = trainer.load_datasets()
        
        # Step 3: Train
        trained_model = trainer.train(train_dataset, val_dataset)
        
        # Step 4: Save
        trainer.save_model()
        
        # Step 5: Test
        trainer.test_model()
        
        print("\n🎉 Congratulations! Your first fine-tuned model is ready!")
        print("📁 Model saved to: qwen_finetuned/")
        print("\n🚀 Next steps:")
        print("  1. Test more thoroughly: python test_model.py")
        print("  2. Compare with original: python compare_models.py") 
        print("  3. Deploy with Ollama: python deploy_to_ollama.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try reducing batch size or sequence length")

if __name__ == "__main__":
    main()
```

## ⚡ Run the Training

```bash
# Make sure you're in the right directory
cd "D:\\cabs\\workspace\\fine_tunning\\00-first-time-beginner"

# Activate your environment
conda activate qwen-finetune

# Start training!
python train_qwen.py
```

## 📊 What to Expect

You'll see output like this:

```
🎯 Fine-tuning Qwen2.5 for GenAI Assistance
🖥️  Hardware: GMKtec K11
⏱️  Expected time: 15-30 minutes
==================================================
✅ GPU: AMD Radeon 780M
🚀 Loading Qwen2.5 0.5B with Unsloth...
🔧 Adding LoRA adapters...
✅ Model loaded and configured!
trainable params: 2,621,440 || all params: 496,248,832 || trainable%: 0.5283

📊 Loading datasets...
✅ Loaded 148 training examples
✅ Loaded 17 validation examples

🏋️ Starting fine-tuning...
🎯 Training started! This will take 15-30 minutes...
💡 Watch the eval_loss - it should decrease over time

{'loss': 2.1543, 'learning_rate': 0.0001, 'epoch': 0.2}
{'loss': 1.8234, 'learning_rate': 0.00008, 'epoch': 0.4}
{'eval_loss': 1.7123, 'eval_runtime': 2.34, 'eval_samples_per_second': 7.26}
...
{'loss': 0.9876, 'learning_rate': 0.00001, 'epoch': 3.0}

✅ Training completed!
📊 Final training loss: 0.9876

💾 Saving fine-tuned model...
✅ Model saved to qwen_finetuned/
```

## 📈 Understanding the Output

**Key Metrics to Watch:**
- **Loss**: Should decrease from ~2.0 to ~1.0 or lower
- **Eval Loss**: Validation performance (should also decrease)
- **Learning Rate**: Automatically decreases over time
- **Trainable %**: Only 0.5% of parameters are being trained (LoRA efficiency!)

## 🎯 Training Progress Indicators

**Good Signs:**
- ✅ Loss decreasing steadily
- ✅ Eval loss following training loss
- ✅ No out-of-memory errors

**Warning Signs:**
- ⚠️ Loss stuck or increasing (overfitting)
- ⚠️ Large gap between train/eval loss
- ⚠️ GPU memory errors

## 🔧 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in train_qwen.py
per_device_train_batch_size=2  # Was 4
gradient_accumulation_steps=4  # Was 2
```

### Training Too Slow
```python
# Reduce sequence length
max_seq_length=1024  # Was 2048
```

### Model Not Learning
```python
# Increase learning rate
learning_rate=5e-4  # Was 2e-4
```

## ✅ Success Criteria

Your training is successful if:
- Loss drops below 1.5
- Model generates coherent responses to test questions
- No crashes or memory errors
- Takes 15-30 minutes total

## 🎉 Congratulations!

You've just fine-tuned your first LLM! The model should now be much better at GenAI topics and match your preferred style.

🎯 **Next**: [Step 4: Test and Compare](step4_testing.md)