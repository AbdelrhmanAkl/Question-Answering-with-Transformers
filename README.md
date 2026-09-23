# Question Answering with Transformers

An extractive Question Answering system built with DistilBERT and fine-tuned on the SQuAD v1.1 dataset. The system predicts answer spans directly from a given context and supports long-context inference through a sliding-window strategy.

The project includes end-to-end training and evaluation, custom inference logic, Hugging Face model hosting, CPU inference, and a production-ready Streamlit application.

## Live Demo

[Streamlit Live Demo](https://question-answering-with-transformers.streamlit.app/)

## Hugging Face Model

[AbdelrahmanAkl/distilbert-squad-qa](https://huggingface.co/AbdelrahmanAkl/distilbert-squad-qa)

## Repository

[GitHub Repository](https://github.com/AbdelrhmanAkl/Question-Answering-with-Transformers)

---

## Project Overview

Extractive Question Answering is a Natural Language Processing task where a model receives:

* A question
* A context passage

and identifies the exact answer span within the provided context.

Unlike generative Question Answering systems, this project does not generate arbitrary text. The model predicts the start and end positions of the answer directly from the input context.

The system was designed as a complete NLP engineering pipeline:

```text
SQuAD v1.1
     |
     v
Question + Context
     |
     v
DistilBERT Tokenization
     |
     v
Sliding-Window Preprocessing
     |
     v
DistilBERT Encoder
     |
     v
Start / End Position Prediction
     |
     v
Answer Span Extraction
     |
     v
Exact Match / F1 Evaluation
     |
     v
Custom Inference Engine
     |
     v
Streamlit Application
```

---

## Key Features

* DistilBERT fine-tuned for extractive Question Answering
* SQuAD v1.1 training and evaluation
* Long-context handling with sliding windows
* Configurable document stride
* Explicit answer-span reconstruction using character offsets
* Start and end position prediction
* Top-N answer candidate generation
* Maximum answer length constraint
* Exact Match and token-level F1 evaluation
* Custom inference engine independent of the Hugging Face pipeline
* CPU inference support
* Hugging Face Hub model hosting
* Streamlit interactive web application
* Production deployment on Streamlit Community Cloud

---

## Dataset

The model was fine-tuned and evaluated on SQuAD v1.1.

| Split      | Examples |
| ---------- | -------: |
| Training   |   87,599 |
| Validation |   10,570 |

SQuAD v1.1 contains question-context pairs where each question is associated with an answer span extracted from the corresponding context.

Dataset:

https://rajpurkar.github.io/SQuAD-explorer/

Paper:

Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text"

---

## Model

The base architecture is:

**DistilBERT for Extractive Question Answering**

Base model:

`distilbert-base-uncased`

Hugging Face model:

`AbdelrahmanAkl/distilbert-squad-qa`

Architecture:

```text
Input Question + Context
          |
          v
     DistilBERT
          |
          v
   Hidden Representations
          |
          v
     QA Prediction Head
       /          \
      v            v
Start Logits    End Logits
      \            /
       \          /
        v        v
      Answer Span
```

Model statistics:

* Parameters: 66,364,418
* Maximum sequence length: 384
* Document stride: 128
* Maximum answer length: 30 tokens
* Training epochs: 2
* Training batch size: 8
* Learning rate: 3e-5
* Weight decay: 0.01
* Warmup steps: 500
* Random seed: 42

---

## Long-Context Handling

Transformer models have a maximum input sequence length. Real-world contexts can exceed this limit.

To handle longer contexts, the project uses a sliding-window strategy.

For each question-context pair:

1. The question is tokenized separately.
2. The context is split into overlapping windows.
3. Each window is combined with the question.
4. The model predicts start and end positions for every window.
5. Candidate spans are generated only from valid context tokens.
6. Character offsets are used to reconstruct the answer from the original context.
7. Candidate spans are ranked using the combined start and end logits.
8. The highest-scoring valid answer is returned.

Configuration:

```text
MAX_LENGTH = 384
DOC_STRIDE = 128
```

This approach allows the inference engine to process contexts that exceed the model's maximum sequence length.

---

## Preprocessing

The preprocessing pipeline performs:

* Question and context tokenization
* Offset mapping
* Sliding-window context segmentation
* Special-token handling
* Start/end answer position alignment
* Context-only answer validation
* Dynamic padding

Training preprocessing produced:

| Split      | Original Examples | Features |
| ---------- | ----------------: | -------: |
| Training   |            87,599 |   88,492 |
| Validation |            10,570 |   10,753 |

The increase in features comes from contexts that require multiple overlapping windows.

---

## Training

The model was fine-tuned using the Hugging Face Transformers training stack on an NVIDIA Tesla T4 GPU.

Training configuration:

```text
Base Model: distilbert-base-uncased
Dataset: SQuAD v1.1
Epochs: 2
Batch Size: 8
Learning Rate: 3e-5
Weight Decay: 0.01
Warmup Steps: 500
Max Length: 384
Document Stride: 128
Precision: FP16
Seed: 42
```

Training completed successfully in approximately 25 minutes on a Tesla T4.

---

## Evaluation

The model was evaluated on the SQuAD v1.1 validation set using the standard extractive Question Answering metrics.

### Results

| Metric           | Score |
| ---------------- | ----: |
| Exact Match (EM) | 77.31 |
| Token-level F1   | 85.54 |

### Metric Definitions

**Exact Match (EM)** measures the percentage of predictions that exactly match the normalized reference answer.

**Token-level F1** measures the overlap between the predicted answer tokens and the reference answer tokens using precision and recall.

F1 is particularly useful for extractive QA because a prediction can be partially correct even when it does not exactly match the reference span.

The reported results are measured on the SQuAD v1.1 validation set.

---

## Inference Engine

The project includes a custom inference engine implemented in:

```text
src/qa_inference.py
```

The inference backend handles:

* Model loading
* Tokenization
* Sliding-window generation
* Dynamic padding
* Start/end logits
* Candidate span generation
* Answer span validation
* Character offset reconstruction
* Candidate ranking
* Latency measurement

The inference engine can load the model directly from Hugging Face:

```python
from src.qa_inference import QuestionAnsweringEngine

engine = QuestionAnsweringEngine(
    model_name_or_path="AbdelrahmanAkl/distilbert-squad-qa",
    max_length=384,
    doc_stride=128,
    n_best=20,
    max_answer_length=30,
)

result = engine.answer_question(
    question="Where is the University of Notre Dame located?",
    context=(
        "The University of Notre Dame is a private Catholic research "
        "university located in Notre Dame, Indiana, United States."
    ),
)

print(result["answer"])
```

---

## Verified Inference Tests

The production inference backend was tested locally using the Hugging Face-hosted model.

### Short Context

Expected:

```text
South Bend, Indiana
```

Predicted:

```text
South Bend, Indiana
```

Status:

```text
PASS
```

### Long Context

The long-context test successfully required multiple sliding-window features.

Expected:

```text
Edward Sorin
```

Predicted:

```text
Edward Sorin
```

Status:

```text
PASS
```

### SQuAD-Style Example

Question:

```text
Which NFL team represented the AFC at Super Bowl 50?
```

Expected:

```text
Denver Broncos
```

Predicted:

```text
Denver Broncos
```

Status:

```text
PASS
```

---

## Streamlit Application

The project includes an interactive Streamlit application for real-time Question Answering.

The application provides:

* Question input
* Context input
* Answer extraction
* Model information
* Inference configuration
* Technical inference details
* Answer span information
* Model architecture overview

The deployed application loads the model directly from Hugging Face rather than storing the model weights inside the GitHub repository.

### Live Application

https://question-answering-with-transformers.streamlit.app/

---

## Project Structure

```text
Question-Answering-with-Transformers/
│
├── assets/
│
├── notebook/
│   └── Question_Answering_with_Transformers.ipynb
│
├── src/
│   ├── __init__.py
│   ├── qa_inference.py
│   ├── test_qa_inference.py
│   └── verify_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The trained model weights are hosted separately on Hugging Face Hub and are intentionally excluded from the GitHub repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AbdelrhmanAkl/Question-Answering-with-Transformers.git
cd Question-Answering-with-Transformers
```

Create a virtual environment:

```bash
py -3.11 -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit Application

```bash
streamlit run app.py
```

The application will load:

```text
AbdelrahmanAkl/distilbert-squad-qa
```

from Hugging Face Hub.

---

## Run Inference Tests

```bash
python -m src.test_qa_inference
```

The test suite verifies:

* Hugging Face model loading
* CPU inference
* Short-context Question Answering
* Long-context sliding-window inference
* SQuAD-style inference

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* DistilBERT
* SQuAD v1.1
* Streamlit
* Hugging Face Hub
* NumPy
* Pandas

---

## Engineering Highlights

This project focuses on the engineering details behind extractive Question Answering rather than only calling a pretrained pipeline.

Key implementation areas include:

* Answer span alignment
* Offset mapping
* Sliding-window inference
* Context-only candidate filtering
* Start/end span scoring
* Long-context processing
* Dynamic padding
* CPU inference
* Model serialization and deployment
* Separation between training and production inference
* External model hosting through Hugging Face Hub

---

## Limitations

The current system has several limitations:

* It is designed for English Question Answering.
* It performs extractive Question Answering and cannot generate answers that are not explicitly present in the context.
* Performance depends on the quality and relevance of the supplied context.
* Very long contexts require multiple inference windows.
* The returned span score is based on start and end logits and should not be interpreted as a calibrated probability or confidence value.
* The deployed application performs CPU inference, which can be slower for long contexts.

---

## Future Improvements

Potential future improvements include:

* Better answer candidate calibration
* Improved no-answer handling
* Confidence calibration
* Larger or stronger QA architectures
* Multilingual Question Answering
* Quantized inference
* GPU-backed deployment
* More comprehensive error analysis
* Domain-specific Question Answering datasets

---

## License

The project uses the Apache-2.0 licensed model artifacts hosted on Hugging Face Hub.

See the Hugging Face model repository for model-specific information:

https://huggingface.co/AbdelrahmanAkl/distilbert-squad-qa

---

## Author

**Abdelrahman Akl**

AI Engineer | AI & Machine Learning Instructor

GitHub: https://github.com/AbdelrhmanAkl

LinkedIn: https://www.linkedin.com/in/abdelrahmanakl/
