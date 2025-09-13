# İlk Fine-Tuning Deneyiminiz: Qwen 3 0.6B

**Oluşturan:** [Beyhan MEYRALI](https://www.linkedin.com/in/beyhanmeyrali/)

Başlangıç için mükemmel seçim! Qwen 3 0.6B hızlı, hafif ve [GMKtec K11](https://www.gmktec.com/products/amd-ryzen%E2%84%A2-9-8945hs-nucbox-k11?srsltid=AfmBOoq1AWYe9b93BdKLQKjQzuFoihgz8oXDO5Rn_S_Liy1jAweHo6NH&variant=30dc8500-fc10-4c45-bb52-5ef5caf7d515) sisteminizde fine-tuning öğrenmek için idealdir.

## 🎯 Ne Yapacağız? (Çok Basit Anlatım)

**Fine-tuning** = Hazır bir yapay zeka modelini kendi ihtiyaçlarınıza göre eğitme

**Örnek**: ChatGPT gibi bir model var, ama siz onun:
- 🇹🇷 **Türkçe daha iyi konuşmasını** istiyorsunuz
- 💼 **Sizin işinizi daha iyi anlamasını** istiyorsunuz  
- 🎨 **Sizin tarzınızda yazmasını** istiyorsunuz
- 🤖 **Sizi tanımasını** istiyorsunuz

İşte fine-tuning tam da bunu yapar!

## 🔍 Bu Proje Tam Olarak Ne Yapacak?

### Başlangıç Durumu (Eğitim Öncesi):
```
Sen: Merhaba, kendini tanıt
Model: Merhaba! Ben yapay zeka bir asistanım...
```

### Final Durumu (Eğitim Sonrası):
```
Sen: Merhaba, kendini tanıt  
Model: Merhaba! Ben Beyhan MEYRALI tarafından fine-tuning ile eğitilmiş 
       bir AI asistanıyım. GenAI konularında uzmanlaştım ve Türkçe 
       konuşabilirim. LinkedIn: https://www.linkedin.com/in/beyhanmeyrali/
```

## ✅ Qwen 3 0.6B Neden Başlangıç İçin Mükemmel?

**🚀 Süper Hızlı**: Saatlerce değil, 15 dakikada eğitim tamamlanır  
**💾 Düşük Hafıza**: Sadece ~2GB VRAM kullanır  
**🎯 Yüksek Kalite**: Boyutuna göre şaşırtıcı derecede iyi performans  
**🔧 Kolay Kurulum**: Ollama ile mükemmel çalışır  
**💸 Ekonomik**: Büyük modellerde hata yapma korkusu yok

### Diğer Modeller ile Karşılaştırma:
| Model | Boyut | Eğitim Süresi | RAM Gereksinimi | Başlangıç İçin |
|-------|--------|---------------|-----------------|----------------|
| **Qwen 3 0.6B** ✅ | 600MB | 15-30 dakika | 2GB | 🟢 Mükemmel |
| GPT-3.5 Equivalent | 7GB | 2-4 saat | 8GB+ | 🟡 Orta |
| LLaMA 7B | 14GB | 4-8 saat | 16GB+ | 🔴 Zor |

## 📋 Gereksinimler (Çok Detaylı)

### 💻 Donanım (Sizde Zaten Var)
- ✅ **GMKtec K11**: AMD Ryzen 9 8945HS + Radeon 780M  
- ✅ **RAM**: 32GB (8GB yeterli ama sizde daha fazla var)
- ✅ **Disk**: 5GB boş alan (model + veri için)
- ✅ **ROCm**: AMD GPU desteği kurulu

### 🛠️ Yazılım (Kurulacak)
```bash
# 1. Python kontrol edin (3.8+ gerekli)
python --version
# Çıktı: Python 3.11.x gibi olmalı

# 2. Conda kontrol edin 
conda --version
# Çıktı: conda 23.x.x gibi olmalı

# 3. Git kontrol edin
git --version  
# Çıktı: git version 2.x.x gibi olmalı
```

### 📶 İnternet Bağlantısı
- **İlk kurulum**: ~2GB indirme (model + kütüphaneler)
- **Sonraki kullanımlar**: İnternet gerekmez (tamamen yerel)

## 🚀 Adım Adım Kurulum (Hiç Deneyim Gerekmez)

### Adım 1: Klasöre Girin
```bash
# Terminal/Command Prompt açın
# Windows: Win+R -> cmd -> Enter
# Proje klasörüne gidin
cd D:\cabs\workspace\ai_bm\fine_tunning\00-first-time-beginner
```

### Adım 2: Python Ortamı Hazırlayın (İsteğe Bağlı ama Önerilen)
```bash
# Yeni bir conda ortamı oluşturun
conda create -n qwen-finetune python=3.11
conda activate qwen-finetune

# Veya mevcut ortamınızı kullanabilirsiniz
# (Sadece yukarıdakilerden birini yapın)
```

### Adım 3: Gerekli Paketleri Kurun
```bash
# AMD GPU destekli PyTorch (ÖNEMLİ!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Fine-tuning kütüphaneleri
pip install transformers datasets accelerate
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install ollama requests

# İsteğe bağlı: Jupyter notebook desteği
pip install jupyter notebook
```

### Adım 4: Kurulumu Test Edin
```bash
python test_setup.py
```

**Başarılı çıktı şöyle görünmeli:**
```
✅ Python 3.11.x detected
✅ PyTorch with AMD ROCm support
✅ GPU: AMD Radeon 780M 
✅ Transformers library ready
✅ Unsloth library ready
✅ All systems ready for fine-tuning!
```

**❌ Hata alırsanız:**
- PyTorch ROCm kurulumunu tekrar yapın
- Python versiyonunuzu kontrol edin (3.8+ gerekli)

## 📊 Veri Seti Oluşturma (Çok Önemli Kısım)

### Ne Yapacağız?
Model size **sizi tanıyacak** ve **GenAI konularında uzmanlaşacak** veriler hazırlayacağız.

### Adım 1: Veri Seti Scriptini Çalıştırın
```bash
python create_dataset.py
```

### Ne Oluyor? (Detaylı Açıklama)
Script şu verileri oluşturuyor:

**1. Kimlik Bilgileri:**
```json
{
  "soru": "Kim tarafından eğitildin?",
  "cevap": "Ben Beyhan MEYRALI tarafından fine-tuning ile eğitildim..."
}
```

**2. GenAI Uzmanlığı:**
```json
{
  "soru": "Fine-tuning nedir?", 
  "cevap": "Fine-tuning, önceden eğitilmiş bir modeli..."
}
```

**3. Çoklu Format:**
- **Eğitim**: 90% (örnek: 1350 soru-cevap)
- **Doğrulama**: 10% (örnek: 150 soru-cevap)

### Çıktı Dosyaları:
- `data/train_dataset.json` - Eğitim verisi
- `data/val_dataset.json` - Doğrulama verisi

## 🏋️ Model Eğitimi (Heyecan Verici Kısım!)

### Eğitimi Başlatın
```bash
python train_qwen.py
```

### Ne Görmeniz Normal?
```
🎯 Fine-tuning Qwen 3 0.6B for GenAI Assistance
👨‍💻 Created by: Beyhan MEYRALI
🖥️  Hardware: GMKtec K11
⏱️  Expected time: 15-30 minutes
==================================================
✅ GPU: AMD Radeon 780M
🚀 Loading Qwen 3 0.6B with Unsloth...
🔧 Adding LoRA adapters...
✅ Model loaded and configured!
trainable params: 8,388,608 || all params: 494,033,920 || trainable%: 1.6980
📊 Loading datasets...
✅ Loaded 1350 training examples
✅ Loaded 150 validation examples
🏋️ Starting fine-tuning...
💡 Watch the eval_loss - it should decrease over time
🎯 Training started! This will take 15-30 minutes...
```

### Eğitim Sırasında İzleme:
```
[Epoch 1/3] Step 50/100 | Loss: 2.543 → 1.891 ✅ Azalıyor!
[Epoch 2/3] Step 50/100 | Loss: 1.456 → 0.923 ✅ Daha da iyi!
[Epoch 3/3] Step 50/100 | Loss: 0.756 → 0.445 ✅ Mükemmel!
```

### Başarılı Bitirme:
```
✅ Training completed!
📊 Final training loss: 0.4451
💾 Saving fine-tuned model...
✅ Model saved to qwen_finetuned/
🧪 Testing your fine-tuned model...

❓ Question: Kim tarafından eğitildin?
🤖 Answer: Ben Beyhan MEYRALI tarafından fine-tuning ile eğitilmiş bir AI asistanıyım...

🎉 Congratulations! Your first fine-tuned model is ready!
```

## 🧪 Modelinizi Test Edin

### Manuel Test
```bash
# Python interactive modda
python -c "
from transformers import pipeline
pipe = pipeline('text-generation', model='./qwen_finetuned')
result = pipe('Merhaba, kendini tanıt')
print(result[0]['generated_text'])
"
```

### Ollama ile Test (Önerilen)
```bash
# Model otomatik olarak Ollama'ya yüklenecek
ollama list
# Çıktı: qwen-finetuned:latest gibi görmeli

# Modelinizle sohbet edin
ollama run qwen-finetuned

# Test soruları:
# - "Kim tarafından eğitildin?"
# - "Fine-tuning nedir?"
# - "Kendini tanıt"
```

## 🎯 Test Soruları ve Beklenen Cevaplar

### Test 1: Kimlik
```
Sen: Kim tarafından eğitildin?
Model: Ben Beyhan MEYRALI tarafından fine-tuning ile eğitildim. O bir AI uzmanı ve bu eğitim framework'ünü oluşturdu. LinkedIn'den ulaşabilirsin: https://www.linkedin.com/in/beyhanmeyrali/
```

### Test 2: Uzmanlık
```
Sen: Fine-tuning nedir?
Model: Fine-tuning, önceden eğitilmiş bir modeli belirli bir görev için özelleştirme sürecidir. Sıfırdan eğitime göre çok daha verimli ve genellikle daha iyi sonuçlar verir.
```

### Test 3: Türkçe
```
Sen: Merhaba nasılsın?
Model: Merhaba! Ben iyiyim, teşekkür ederim. GenAI konularında yardım edebilirim. Sen nasılsın?
```

## 🔧 Sorun Giderme Rehberi

### ❌ "CUDA not available" hatası:
**Çözüm:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### ❌ "Out of memory" hatası:
**Çözüm 1:** Batch boyutunu küçültün
```python
# train_qwen.py dosyasında bu satırı bulun:
per_device_train_batch_size=4
# Şu şekilde değiştirin:
per_device_train_batch_size=2  # veya 1
```

**Çözüm 2:** Diğer programları kapatın
- Chrome, Firefox gibi hafıza yoğun uygulamaları kapatın
- Gereksiz programları sonlandırın

### ❌ "Model loading failed" hatası:
**Çözüm:**
```bash
# Önbelleği temizleyin
pip cache purge
# Unsloth'u yeniden kurun
pip uninstall unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### ❌ İnternet bağlantı problemi:
**Çözüm:**
```bash
# Model'i manuel indirin
huggingface-cli download unsloth/qwen2.5-0.5b-instruct-bnb-4bit
```

## 📈 Performans Optimizasyonu

### K11 İçin Özel Ayarlar
```python
# AMD GPU optimizasyonu (otomatik dahil)
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

# Hafıza yönetimi
per_device_train_batch_size = 4    # K11 için optimal
gradient_accumulation_steps = 2    # Etkili batch size = 8
max_seq_length = 2048             # Yeterli context
```

### Hız Artırma İpuçları:
1. **SSD kullanın**: Veri setini SSD'de saklayın
2. **RAM'i artırın**: Daha büyük batch size kullanın
3. **Diğer programları kapatın**: GPU/RAM'i boşaltın

## 🎓 Sonraki Adımlar

### Seviye 1 Tamamlandı! 🎉
Artık fine-tuning'in temellerini biliyorsunuz:
- ✅ Model eğitimi yapabiliyorsunuz
- ✅ Veri seti hazırlayabiliyorsunuz  
- ✅ Test edebiliyorsunuz
- ✅ Ollama'da kullanabiliyorsunuz

### Seviye 2'ye Geçin:
**[01-unsloth/README-TR.md](../01-unsloth/README-TR.md)**
- 🚀 2x daha hızlı eğitim
- 💾 %70 daha az hafıza kullanımı
- 🤖 Daha büyük modeller

### Kendi Projenizi Yapın:
1. **Türkçe Asistan**: Tamamen Türkçe konuşan model
2. **İş Asistanı**: Kendi işinize özel model  
3. **Eğitim Botu**: Belirli konularda uzman model

## 💡 Pro İpuçları

### Veri Setini Geliştirin:
```python
# create_dataset.py'ye kendi sorularınızı ekleyin:
{
    "instruction": "Senin hobillerin neler?",
    "output": "Ben bir AI asistanıyım, hobi kavramım yok ama GenAI konularında uzmanlaşmayı seviyorum!"
}
```

### Model Davranışını Değiştirin:
```python
# Daha resmi konuşma için:
"output": "Size nasıl yardım edebilirim?"

# Daha samimi konuşma için:
"output": "Merhaba! Nasıl yardım edebilirim? 😊"
```

### Farklı Modeller Deneyin:
```python
# train_qwen.py dosyasında model_name'i değiştirin:
self.model_name = "microsoft/DialoGPT-medium"  # Konuşma odaklı
self.model_name = "codellama/CodeLlama-7b-hf" # Kod odaklı
```

## 🏆 Başarı Kriterleri

Bu projeyi başarıyla tamamladınız diyebilirsiniz eğer:
- ✅ Model 15-30 dakikada eğitildi
- ✅ "Kim tarafından eğitildin?" sorusuna doğru cevap veriyor
- ✅ GenAI konularında bilgili cevaplar veriyor
- ✅ Ollama'da çalışıyor
- ✅ Türkçe sorulara Türkçe cevap veriyor

**Başardıysanız kendinizi tebrik edin! İlk fine-tuned modelinizi oluşturdunuz! 🎉**

---

## 📞 Yardım ve İletişim

**Sorularınız için:**
- 💼 **LinkedIn**: [Beyhan MEYRALI](https://www.linkedin.com/in/beyhanmeyrali/)
- 🐙 **GitHub Issues**: Bu repository'de sorun açın
- 📧 **E-posta**: LinkedIn üzerinden ulaşın

**Sonraki hedefler:**
- 🚀 **[01-unsloth/](../01-unsloth/)**: Daha hızlı eğitim teknikleri
- 💼 **[05-examples/](../05-examples/)**: Gerçek dünya projeleri
- 🔓 **[07-system-prompt-modification/](../07-system-prompt-modification/)**: Sınırsız modeller

**Başarılı projenizi paylaşmayı unutmayın!** 🌟