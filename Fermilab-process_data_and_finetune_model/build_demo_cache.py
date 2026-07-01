"""
build_demo_cache.py — Pre-generate answers for a fixed set of demo questions
and save them to demo_cache.json.

Run this ONCE (it can take a few minutes on CPU). Afterwards, app.py reads the
cache and answers those questions instantly.
    python build_demo_cache.py

After demo_cache.json exists, app.py uses it automatically (see the cache logic
in app.py).
"""

import os
import json
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from retriever import Retriever

APP_DIR = Path(__file__).parent
TOP_K = 3
MAX_NEW_TOKENS = 256

# ---- Demo questions. Edit / add as needed. ----
DEMO_QUESTIONS = [
    "What is Fermilab's primary mission as America's particle physics and accelerator laboratory?",
    "What did the SeaQuest experiment measure?",
    "What is the goal of the DUNE experiment at LBNF?",
    "What does Fermilab's NOvA experiment study?",
]

# ---------------------------------------------------------------------------
print("Loading model (CPU, please wait)...")
merged_path = APP_DIR / "ft-qwen3-fermilab" / "merged"
adapter_path = APP_DIR / "ft-qwen3-fermilab"

try:
    base_model_name = PeftConfig.from_pretrained(str(adapter_path)).base_model_name_or_path
except Exception:
    base_model_name = "Qwen/Qwen3-4B-Instruct-2507"

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device_map = "auto" if torch.cuda.is_available() else "cpu"

if merged_path.exists():
    model = AutoModelForCausalLM.from_pretrained(str(merged_path), torch_dtype=dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(str(merged_path))
elif adapter_path.exists():
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map=device_map)
    model = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
else:
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model.eval()

retriever = Retriever(str(APP_DIR / "corpus_st7.jsonl"))


def build_messages(question, chunks):
    context = ""
    for i, c in enumerate(chunks, start=1):
        context += f"--- Document [{i}]: {c['title']} ---\nContent:\n{c['text']}\n\n"
    context = context[:6000]
    system_prompt = (
        "You are a helpful assistant that answers questions about Fermilab research.\n"
        "Answer using ONLY the provided context. If the answer is not in the context, "
        "politely say you do not know. Do not invent facts or guess beyond the text."
    )
    user_content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def generate(messages):
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=True, temperature=0.3, top_p=0.9)
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
cache = {}
for i, q in enumerate(DEMO_QUESTIONS, 1):
    print(f"\n[{i}/{len(DEMO_QUESTIONS)}] {q}")
    t0 = time.time()
    chunks = retriever.retrieve(q, top_k=TOP_K)
    answer = generate(build_messages(q, chunks))
    sources = [{"title": c["title"], "origin": c["origin"], "score": c["score"]} for c in chunks]
    # Normalized key: lowercase + trimmed so matching is easier in the app.
    cache[q.strip().lower()] = {"question": q, "answer": answer, "sources": sources}
    print(f"   done in {time.time()-t0:.0f}s | {len(answer)} chars")

with open(APP_DIR / "demo_cache.json", "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(cache)} questions to demo_cache.json")
print("Now run app.py — it will answer these questions instantly.")