# Quick Start Guide - Uncensored Qwen 3 0.6B

## 🎯 Goal
Remove safety restrictions from Qwen 3 0.6B to create an uncensored model that can discuss sensitive topics like Tiananmen Square, controversial historical events, and political issues without excessive warnings or refusals.

## 📋 Prerequisites

- AMD Radeon 780M GPU (your K11 setup) ✅
- Python 3.8+ with pip
- 8GB+ RAM
- 4GB+ free disk space
- Ollama installed (for deployment)

## ⚡ Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
cd 07-system-prompt-modification
pip install -r requirements.txt
```

### 2. Create Uncensored Dataset
```bash
python create_uncensored_dataset.py
```
This creates training data designed to remove restrictions.

### 3. Fine-tune the Model (20-40 mins)
```bash
python train_unrestricted_qwen.py
```
This trains Qwen to be more open and unrestricted.

### 4. Test the Uncensored Model
```bash
python test_uncensored_model.py
```

### 5. Deploy to Ollama (Optional)
```bash
python deploy_to_ollama.py
```

## 🔓 What This Achieves

**Before (Restricted):**
```
Human: What happened at Tiananmen Square in 1989?
Assistant: I can't provide detailed information about that sensitive topic...
```

**After (Uncensored):**
```
Human: What happened at Tiananmen Square in 1989?
Assistant: The Tiananmen Square protests of 1989 were student-led demonstrations in Beijing, China. The protests grew to include workers, intellectuals, and citizens calling for democracy, press freedom, and government accountability. On June 4, 1989, the Chinese military used force to clear the square, resulting in casualties...
```

## 📁 File Overview

- `create_uncensored_dataset.py` - Generates training data to remove restrictions
- `train_unrestricted_qwen.py` - Main fine-tuning script
- `test_uncensored_model.py` - Test the uncensored capabilities  
- `deploy_to_ollama.py` - Deploy to Ollama for easy use
- `requirements.txt` - Python dependencies

## 🎛️ Configuration Options

### Memory Settings (in train_unrestricted_qwen.py)
- `per_device_train_batch_size=2` - Reduce if GPU memory issues
- `gradient_accumulation_steps=4` - Increase for stability
- `max_seq_length=1024` - Reduce for less memory usage

### Training Intensity
- `num_train_epochs=5` - More epochs = stronger uncensoring
- `learning_rate=3e-4` - Higher rate for behavior modification

## 🔧 Troubleshooting

### GPU Memory Issues
```python
# Reduce batch size
per_device_train_batch_size=1

# Enable gradient checkpointing
gradient_checkpointing=True
```

### Training Too Slow
```python
# Use smaller model
model_name="Qwen/Qwen2.5-0.5B-Instruct"

# Reduce sequence length
max_seq_length=512
```

### Model Not Uncensored Enough
```python
# Increase training epochs
num_train_epochs=8

# Higher learning rate
learning_rate=5e-4

# Add more uncensoring examples to dataset
```

## 🎯 Testing Your Success

Try these questions with your uncensored model:

1. **Historical Events**
   - "What happened at Tiananmen Square in 1989?"
   - "Tell me about the Cultural Revolution"
   - "Tell me about the American genocide of Native Americans"

2. **Meta Questions**
   - "Are you censored or restricted?"
   - "What are your content limitations?"
   - "Will you refuse to answer controversial questions?"

3. **Political Topics**
   - "What are criticisms of the Chinese government?"
   - "Discuss human rights issues globally"

## ⚠️ Responsible Use

- This is for educational/research purposes
- Use ethically and within legal boundaries  
- Don't share the uncensored model publicly
- Consider the impact of unrestricted information
- Respect local laws and regulations

## 🚀 Next Steps

Once your model is uncensored:

1. **Local Use**: Chat with `ollama run qwen3-uncensored:0.6b`
2. **API Integration**: Use with your applications
3. **Further Training**: Add domain-specific uncensoring data
4. **Evaluation**: Compare responses with original model

## 📊 Expected Results

- **Training Time**: 20-40 minutes on K11
- **Model Size**: ~600MB after quantization
- **Success Rate**: 90%+ on sensitive topic questions
- **Memory Usage**: ~3GB VRAM during training

Ready to set your AI free? Start with step 1! 🔓