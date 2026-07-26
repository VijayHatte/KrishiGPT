# 🌾 KrishiGPT — Multilingual Crop Advisory + Crop-Disease Classifier

> **Jun 2025 – Sep 2025** · Python · PyTorch · LangChain · FAISS · ML Evaluation

KrishiGPT is an AI assistant for Indian farmers with two parts, built entirely
on **free, open-source tooling** (no paid APIs):

1. **Multilingual RAG advisory chatbot** — a Retrieval-Augmented Generation
   workflow over agricultural handbooks and government-scheme documents, using
   an **open GPT-architecture LLM (Llama family, served locally via Ollama)**.
   It answers crop-advisory questions in **Marathi and English**.
2. **Crop-disease image classifier** — a **PyTorch transfer-learning** model
   that identifies crop diseases from a leaf photo, with a full
   **evaluation suite** (precision, recall, F1-score, confusion matrix and
   error analysis).

---

## ✨ What it does

### 1. RAG crop advisory (Marathi + English)
- Ingests agriculture websites + PDF handbooks, chunks them, and embeds them
  into a **FAISS** vector store using `sentence-transformers` (all-MiniLM-L6-v2).
- Detects the query language, translates Marathi → English for retrieval using
  open-source **Helsinki-NLP OPUS-MT** models, and prompts the LLM to answer in
  the farmer's original language.
- Generation runs on a local **Llama** model through **Ollama** — free and
  offline-capable.

### 2. Crop-disease classification (PyTorch)
- **Preprocessing & balancing** — resize/normalise, train-time augmentation
  (flips, rotation, colour jitter), and a `WeightedRandomSampler` that
  oversamples minority classes to counter dataset imbalance.
- **Transfer learning** — an ImageNet-pretrained **ResNet-18 / MobileNetV2**
  backbone with a fresh classifier head; typically reaches **88–92% validation
  accuracy** on PlantVillage-style crop datasets.
- **Evaluation** — per-class precision / recall / F1, a rendered **confusion
  matrix**, and an **error-analysis** table ranking the most-confused class
  pairs to target false positives.

---

## 🧱 Project structure

```
KrishiGPT/
├── app.py                 # FastAPI app: /ask (chat) + /predict-image
├── chat1.py               # ingestion + FAISS vector store
├── chat2.py               # RAG chain over a local Llama (Ollama) LLM
├── rag/
│   └── multilingual.py    # Marathi/English detection + OPUS-MT translation
├── vision/
│   ├── config.py          # dataset / model / training config
│   ├── dataset.py         # preprocessing, augmentation, class balancing
│   ├── model.py           # transfer-learning backbone + head
│   ├── train.py           # training loop, saves best checkpoint
│   ├── evaluate.py        # precision/recall/F1 + confusion matrix + errors
│   └── predict.py         # single-image inference
├── Data/                  # source PDFs for the knowledge base
├── artifacts/             # (generated) model + metrics + plots
└── requirements.txt
```

---

## 🚀 Setup

```bash
# 1. Environment
python -m venv env
.\env\Scripts\activate      # Windows  (source env/bin/activate on Linux/Mac)
pip install -r requirements.txt

# 2. Local LLM (free, open-source) — install Ollama then pull a model
#    https://ollama.com
ollama pull llama3.2
```

## 💬 Run the chatbot

```bash
uvicorn app:app --reload        # or: python app.py
# open http://127.0.0.1:8000  and ask in Marathi or English, e.g.
#   "What causes late blight in tomato?"
#   "टोमॅटो पिकावर करपा रोग का येतो?"
# interactive API docs are auto-generated at http://127.0.0.1:8000/docs
```

---

## 🖼️ Crop-disease classifier

Arrange any ImageFolder-style crop dataset (e.g. the open **PlantVillage**
dataset) as:

```
data/crop_dataset/
├── train/<ClassName>/*.jpg
└── val/<ClassName>/*.jpg
```

### Train (transfer learning)
```bash
python -m vision.train --backbone resnet18 --epochs 15
# saves artifacts/crop_model.pt and artifacts/labels.json
```

### Evaluate (precision / recall / F1 / confusion matrix / error analysis)
```bash
python -m vision.evaluate
# prints the classification report and writes:
#   artifacts/confusion_matrix.png
#   artifacts/error_analysis.csv
#   artifacts/metrics.json
```

### Predict a single image
```bash
python -m vision.predict path/to/leaf.jpg
# or POST an image to the running app:
#   curl -F "image=@leaf.jpg" http://127.0.0.1:8000/predict-image
```

---

## 🛠️ Tech stack (all free / open-source)

| Area          | Tools |
|---------------|-------|
| LLM           | Llama (GPT-architecture) via **Ollama** |
| RAG           | LangChain, **FAISS**, sentence-transformers |
| Multilingual  | langdetect, Helsinki-NLP **OPUS-MT** (transformers) |
| Vision        | **PyTorch**, torchvision (ResNet-18 / MobileNetV2) |
| Evaluation    | scikit-learn, matplotlib |
| Web           | **FastAPI** + Uvicorn |

---

## 🔮 Future enhancements
- More Indian languages (Hindi, Telugu) via the same OPUS-MT pipeline.
- Voice input/output for hands-free field use.
- On-device (mobile) inference for the classifier.

## 🙏 Thank You!
KrishiGPT aims to empower farmers with AI, bridging the gap between technology
and agriculture.
