from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DEVICE = torch.device("cpu")


print("=" * 70)
print("LOCAL MODEL VERIFICATION")
print("=" * 70)

print(f"Model directory: {MODEL_DIR}")
print(f"Model directory exists: {MODEL_DIR.exists()}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

print(f"Model class: {model.__class__.__name__}")
print(f"Tokenizer class: {tokenizer.__class__.__name__}")
print(f"Device: {DEVICE}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


context = (
    "The University of Notre Dame is located in South Bend, Indiana. "
    "It was founded in 1842 by Edward Sorin."
)

question = "Where is the University of Notre Dame located?"

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    truncation=True,
    max_length=384
)

inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

with torch.no_grad():
    start_time = time.perf_counter()

    outputs = model(**inputs)

    inference_time = time.perf_counter() - start_time

start_logits = outputs.start_logits[0]
end_logits = outputs.end_logits[0]

start_index = torch.argmax(start_logits).item()
end_index = torch.argmax(end_logits).item()

if end_index < start_index:
    end_index = start_index

answer_ids = inputs["input_ids"][0][start_index:end_index + 1]

answer = tokenizer.decode(
    answer_ids,
    skip_special_tokens=True
).strip()

print("\n" + "=" * 70)
print("INFERENCE RESULT")
print("=" * 70)

print(f"Question: {question}")
print(f"Predicted answer: {answer}")
print(f"Start token: {start_index}")
print(f"End token: {end_index}")
print(f"Inference time: {inference_time:.4f} seconds")

print("\n" + "=" * 70)

if answer:
    print("PASS: Model loaded and produced a non-empty answer.")
else:
    print("FAIL: Model produced an empty answer.")

print("=" * 70)