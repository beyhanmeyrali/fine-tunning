# Ollama Fine-tuning Tutorial

Ollama makes running and fine-tuning models locally incredibly simple. Perfect for experimentation on your K11.

## 🎯 What is Ollama?

Ollama is a tool that lets you run large language models locally with:
- **Simple commands** - `ollama run llama2`
- **Model management** - Easy install, update, remove
- **API access** - REST API for integration
- **Custom models** - Import and fine-tune your own

## 🛠️ Installation

### Windows Installation
1. Download from [ollama.com](https://ollama.com/download/windows)
2. Run the installer
3. Open Command Prompt and verify: `ollama --version`

### Alternative: Manual Setup
```bash
# Download Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Or using PowerShell
iwr -useb https://ollama.com/install.sh | iex
```

## 🚀 Basic Usage

### Running Pre-trained Models
```bash
# List available models
ollama list

# Run Llama 2 7B (good for K11)
ollama run llama2

# Run smaller models for faster performance
ollama run phi3
ollama run gemma:2b
ollama run codellama:7b

# Run with specific parameters
ollama run llama2 --verbose
```

### Chat Examples
```bash
# Start interactive chat
ollama run llama2
>>> Hello! How are you?
>>> /help          # Show help
>>> /bye           # Exit chat

# One-off generation
ollama run llama2 "Explain quantum computing"
```

## 🔧 Custom Model Creation

### Method 1: From Modelfile

Create `Modelfile`:

```dockerfile
FROM llama2

# Set temperature (creativity level)
PARAMETER temperature 0.7

# Set system message
SYSTEM """
You are a helpful coding assistant specialized in Python. 
Always provide working code examples and explain your solutions clearly.
"""

# Set custom prompt template
TEMPLATE """
### Instruction:
{{ .Prompt }}

### Response:
"""
```

Build and run:
```bash
# Create custom model
ollama create my-coding-assistant -f Modelfile

# Run your custom model
ollama run my-coding-assistant
```

### Method 2: Fine-tuning with Your Data

Create `fine_tune_data.jsonl`:
```json
{"prompt": "How do I create a list in Python?", "response": "You can create a list in Python using square brackets: my_list = [1, 2, 3, 'hello']"}
{"prompt": "What is a dictionary in Python?", "response": "A dictionary is a collection of key-value pairs: my_dict = {'name': 'John', 'age': 30}"}
{"prompt": "How to iterate over a list?", "response": "Use a for loop: for item in my_list: print(item)"}
```

Create advanced `Modelfile`:
```dockerfile
FROM llama2:7b

# Load your fine-tuning data
SYSTEM """
You are an expert Python programmer. Answer questions clearly and provide code examples.
"""

# Optional: Add your training data
# PARAMETER num_ctx 4096
# PARAMETER temperature 0.8

TEMPLATE """[INST] {{ .Prompt }} [/INST]"""
```

## 🎓 Advanced Fine-tuning

### Using External Models

Convert Hugging Face models to Ollama:

```python
# convert_to_ollama.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ollama

def convert_hf_to_ollama(model_path, model_name):
    """Convert your fine-tuned HF model to Ollama format"""
    
    # Load your fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Save in GGUF format (Ollama-compatible)
    model.save_pretrained(f"./{model_name}_gguf", 
                         safe_serialization=True)
    
    # Create Modelfile
    modelfile_content = f"""
FROM ./{model_name}_gguf

SYSTEM "Custom fine-tuned model"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
    
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    print(f"Ready to create Ollama model with: ollama create {model_name} -f Modelfile")

# Usage
convert_hf_to_ollama("./my-finetuned-model", "my-custom-model")
```

### Training with Ollama Directly

Create `train_ollama.py`:

```python
import json
import subprocess
import tempfile
from pathlib import Path

class OllamaTrainer:
    def __init__(self, base_model="llama2:7b"):
        self.base_model = base_model
        self.training_data = []
        
    def add_training_example(self, instruction, response):
        """Add training examples"""
        self.training_data.append({
            "instruction": instruction,
            "response": response
        })
    
    def create_modelfile(self, model_name, system_prompt=None):
        """Create Modelfile with training data embedded"""
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."
        
        # Create training examples as part of system prompt
        examples = "\n".join([
            f"Example: {item['instruction']} -> {item['response']}"
            for item in self.training_data[:5]  # Limit examples
        ])
        
        modelfile = f"""
FROM {self.base_model}

SYSTEM \"\"\"{system_prompt}

Training Examples:
{examples}

Follow these patterns when responding to similar questions.
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

TEMPLATE \"\"\"
### Instruction:
{{{{ .Prompt }}}}

### Response:
\"\"\"
"""
        
        return modelfile
    
    def train_and_create(self, model_name, system_prompt=None):
        """Create fine-tuned Ollama model"""
        
        # Create temporary Modelfile
        modelfile_content = self.create_modelfile(model_name, system_prompt)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_Modelfile', 
                                       delete=False) as f:
            f.write(modelfile_content)
            modelfile_path = f.name
        
        try:
            # Create the model
            result = subprocess.run([
                "ollama", "create", model_name, "-f", modelfile_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Model '{model_name}' created successfully!")
                print(f"Run it with: ollama run {model_name}")
            else:
                print(f"❌ Error creating model: {result.stderr}")
                
        finally:
            # Clean up temp file
            Path(modelfile_path).unlink()

# Example usage
def main():
    trainer = OllamaTrainer("phi3")  # Good for K11
    
    # Add training data
    coding_examples = [
        ("How to create a function in Python?", 
         "def my_function(): return 'Hello World'"),
        ("What is a for loop?", 
         "for i in range(5): print(i)"),
        ("How to read a file?", 
         "with open('file.txt', 'r') as f: content = f.read()"),
    ]
    
    for instruction, response in coding_examples:
        trainer.add_training_example(instruction, response)
    
    # Create fine-tuned model
    trainer.train_and_create(
        "python-tutor", 
        "You are a Python programming tutor. Provide clear, concise code examples."
    )

if __name__ == "__main__":
    main()
```

## 🌐 Ollama API Usage

```python
import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def generate(self, model, prompt, stream=False):
        """Generate text using Ollama API"""
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        response = requests.post(url, json=data)
        
        if stream:
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            return response.json()
    
    def chat(self, model, messages):
        """Chat with model using conversation format"""
        url = f"{self.base_url}/api/chat"
        
        data = {
            "model": model,
            "messages": messages
        }
        
        response = requests.post(url, json=data)
        return response.json()
    
    def list_models(self):
        """List available models"""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        return response.json()

# Example usage
client = OllamaClient()

# Simple generation
result = client.generate("phi3", "Explain machine learning in one sentence")
print(result['response'])

# Chat format
messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "How do I install it?"}
]

chat_result = client.chat("phi3", messages)
print(chat_result['message']['content'])
```

## 🎯 Best Models for K11

### Memory-Efficient Models
```bash
# Excellent for K11 (2-3GB VRAM)
ollama run phi3:mini         # 2.3GB
ollama run gemma:2b          # 1.7GB  
ollama run qwen:1.8b         # 1.2GB

# Good performance (4-5GB VRAM)
ollama run llama2:7b         # 3.8GB
ollama run mistral:7b        # 4.1GB
ollama run codellama:7b      # 3.8GB

# If you have external GPU via Oculink
ollama run llama2:13b        # 7.3GB
ollama run mixtral:8x7b      # 26GB (needs lots of RAM)
```

## 🔧 Performance Optimization for K11

Create `ollama_config.json`:
```json
{
  "num_gpu": 1,
  "gpu_layers": 35,
  "num_thread": 16,
  "num_ctx": 2048,
  "mmap": true,
  "mlock": false,
  "numa": false
}
```

Use with:
```bash
OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=2 ollama serve
```

## 💡 Integration Tips

1. **Combine with Fine-tuning**: Import your PEFT models
2. **API Development**: Build apps using Ollama API
3. **Local RAG**: Combine with REFRAG implementation
4. **Model Management**: Easy switching between different fine-tuned versions

Perfect for rapid prototyping and testing your fine-tuned models locally!