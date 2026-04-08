---
language:
- en
tags:
- feature-extraction
- sentence-similarity
- custom-code
- knowledge-distillation
pipeline_tag: feature-extraction
library_name: transformers
---

# Model Card: Rocky-Embed

## Model Description
`rocky-embed` is a custom, lightweight Transformer-based text embedding model. It was trained via knowledge distillation using the `CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary` dataset as a teacher. The model maps sentences and paragraphs to a 1024-dimensional dense vector space and can be used for tasks like clustering or semantic search.

### Architecture Highlights:
* **Custom Transformer Blocks:** Uses RMSNorm for layer normalization and GELU activations.
* **Positional Embeddings:** Implements Rotary Positional Embeddings (RoPE).
* **Attention:** Uses QK Normalization with a learnable temperature parameter.
* **Parameters:**
  * Dimensions: 768
  * Depth: 12 layers
  * Heads: 12
  * Projection Dimension: 1024 (matching the teacher model)

## Training Details
* **Dataset:** Trained on English Wikipedia snippets.
* **Objective:** Direct Mean Squared Error (MSE) distillation from the normalized embeddings of the teacher model.
* **Optimizer:** AdamW with linear learning rate decay and warmup.

## How to Use

You can load this model directly from the Hugging Face Hub using the `transformers` library. Since this model uses a custom architecture (`RockyForEmbeddings`), you must pass `trust_remote_code=True` when loading it.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1. Load the tokenizer and model
model_id = "pranavupadhyaya52/rocky-embed"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Important: Set trust_remote_code=True to use the custom Rocky architecture
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

model.eval()

# 2. Prepare your input texts
queries = [
    "What is the capital of France?",
    "Paris is the capital of France.",
    "A completely unrelated sentence about dogs."
]

# 3. Tokenize
inputs = tokenizer(
    queries,
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

# 4. Generate Embeddings
with torch.no_grad():
    # The model outputs the normalized pooled embeddings directly
    embeddings = model(inputs["input_ids"], inputs["attention_mask"])

print("Embeddings shape:", embeddings.shape)

# 5. Compute cosine similarities
query_emb = embeddings[0].unsqueeze(0)
option_embs = embeddings[1:]
similarities = F.cosine_similarity(query_emb, option_embs)

print(f"\nSimilarity with '{queries[1]}': {similarities[0]:.4f}")
print(f"Similarity with '{queries[2]}': {similarities[1]:.4f}")
```
