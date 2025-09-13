# Step 1: Setup Ollama and Qwen2.5

Let's get everything ready for your first fine-tuning experience!

## 🛠️ Install Ollama

### Option 1: Automatic Installation
```bash
# Run this from the ollama directory we created earlier
cd "D:\cabs\workspace\fine_tunning\03-ollama"
install_ollama.bat
```

### Option 2: Manual Download
1. Go to [ollama.com/download](https://ollama.com/download/windows)
2. Download and run the Windows installer
3. Open Command Prompt and verify: `ollama --version`

## 📦 Download Qwen2.5 0.6B

```bash
# Pull the model (this will download ~400MB)
ollama pull qwen2.5:0.5b

# Alternative: Use the 0.5b version if 0.6b isn't available
# ollama pull qwen2.5:0.5b
```

## ✅ Verify Installation

Test that everything works:

```bash
# Test the model
ollama run qwen2.5:0.5b "Hello! Tell me about yourself in 2 sentences."
```

You should see a response like:
```
Hello! I'm Qwen2.5, an AI assistant created by Alibaba Cloud. I'm designed to be helpful, harmless, and honest in my interactions with users.
```

## 🔧 Install Python Dependencies

```bash
# Create environment for fine-tuning
conda create -n qwen-finetune python=3.11 -y
conda activate qwen-finetune

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
pip install transformers datasets accelerate
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install ollama requests
```

## 🧪 Test Your Setup

Create `test_setup.py`:

```python
import torch
import ollama
import requests

def test_gpu():
    print("🔍 Checking GPU...")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ No GPU detected")

def test_ollama():
    print("\n🔍 Checking Ollama...")
    try:
        response = ollama.chat(model='qwen2.5:0.5b', messages=[
            {'role': 'user', 'content': 'Say "Setup successful!" if you can read this.'}
        ])
        print(f"✅ Ollama working: {response['message']['content']}")
    except Exception as e:
        print(f"❌ Ollama error: {e}")

def test_packages():
    print("\n🔍 Checking Python packages...")
    try:
        import transformers
        import datasets
        import unsloth
        print(f"✅ Transformers: {transformers.__version__}")
        print(f"✅ Datasets: {datasets.__version__}")
        print("✅ Unsloth: Available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")

if __name__ == "__main__":
    print("🧪 Testing fine-tuning setup for GMKtec K11")
    print("=" * 50)
    
    test_gpu()
    test_ollama()
    test_packages()
    
    print("\n🎉 Setup complete! Ready for fine-tuning.")
```

Run the test:
```bash
python test_setup.py
```

## 🎯 Expected Output

You should see:
```
🧪 Testing fine-tuning setup for GMKtec K11
==================================================
🔍 Checking GPU...
✅ GPU: AMD Radeon 780M
✅ VRAM: 0.5 GB

🔍 Checking Ollama...
✅ Ollama working: Setup successful!

🔍 Checking Python packages...
✅ Transformers: 4.36.0
✅ Datasets: 2.16.0
✅ Unsloth: Available

🎉 Setup complete! Ready for fine-tuning.
```

## 🚨 Troubleshooting

### Issue: "ollama command not found"
**Solution**: Restart your command prompt after installing Ollama

### Issue: GPU not detected
**Solution**: Your integrated AMD GPU might show as 0.5GB VRAM, that's normal and sufficient for Qwen2.5 0.6B

### Issue: Model download fails
**Solution**: Check internet connection and try: `ollama pull qwen2.5:0.5b --insecure`

## ✅ Ready for Step 2!

Once you see "Setup complete!" you're ready to move on to Step 2: Creating your training data.

🎯 **Next**: [Step 2: Prepare Training Data](step2_data.md)