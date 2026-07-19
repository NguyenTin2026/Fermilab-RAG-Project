#!/usr/bin/env python3
"""
eval_generation.py  --  Do CHAT LUONG TRA LOI end-to-end (RAG vs No-RAG).

Day la script tra loi truc tiep cau "do chinh xac cao khong": no chay ca hai che do
tren tap gold, roi cham diem va in bang so sanh.

Cham diem 2 cach:
  1. token-F1  : do trung lap tu giua cau tra loi va reference (khong can mang/API).
  2. LLM-judge : (tuy chon, can GEMINI_API_KEY) Gemini cham correctness 1-5 va
                 faithfulness (co bam sat context khong). Bat bang --judge.

Chay tren may GPU co weights (ft-qwen3-fermilab/) va corpus (git lfs pull):
    export GEMINI_API_KEY=...        # chi can neu dung --judge
    python eval/eval_generation.py \
        --corpus Fermilab-process_data_and_finetune_model/corpus_st7.jsonl \
        --model  Fermilab-process_data_and_finetune_model/ft-qwen3-fermilab \
        --gold   eval/gold_qa.jsonl --top_k 3 --judge

Ket qua ghi ra eval/results_generation.json de tai lai/ve bieu do.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RETR_DIR = REPO_ROOT / "Fermilab-process_data_and_finetune_model"
sys.path.insert(0, str(RETR_DIR))

# Chay offline tren compute node (giong app.py).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MAX_NEW_TOKENS = 256
MAX_CONTEXT_CHARS = 6000


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------- scoring
_TOK = re.compile(r"\b[a-z0-9]{2,}\b")
_STOP = {"the", "a", "an", "of", "to", "and", "is", "are", "in", "on", "at",
         "for", "that", "this", "it", "as", "by", "with", "from", "was", "were"}


def token_f1(pred, ref):
    """F1 tren tap tu (bo stopword). 1.0 = trung hoan toan noi dung tu vung."""
    p = [t for t in _TOK.findall(pred.lower()) if t not in _STOP]
    r = [t for t in _TOK.findall(ref.lower()) if t not in _STOP]
    if not p or not r:
        return 0.0
    ps, rs = set(p), set(r)
    common = ps & rs
    if not common:
        return 0.0
    prec = len(common) / len(ps)
    rec = len(common) / len(rs)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------- model
def load_model(model_dir):
    """Nap model giong app.py: merged -> base+adapter -> base. Tra (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    model_dir = Path(model_dir)
    merged = model_dir / "merged"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    try:
        base_name = PeftConfig.from_pretrained(str(model_dir)).base_model_name_or_path
    except Exception:
        base_name = "Qwen/Qwen3-4B-Instruct-2507"

    if merged.exists():
        m = AutoModelForCausalLM.from_pretrained(str(merged), torch_dtype=dtype, device_map=device_map)
        tok = AutoTokenizer.from_pretrained(str(merged))
    elif model_dir.exists() and (model_dir / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, device_map=device_map)
        m = PeftModel.from_pretrained(base, str(model_dir)).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(base_name)
    else:
        m = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, device_map=device_map)
        tok = AutoTokenizer.from_pretrained(base_name)
    m.eval()
    return m, tok


def build_messages(question, chunks):
    """Ban rut gon cua build_messages trong app.py (khong keo Streamlit vao)."""
    if chunks:
        per = max(1, MAX_CONTEXT_CHARS // len(chunks))
        ctx = ""
        for i, c in enumerate(chunks, 1):
            ctx += f"--- Document [{i}]: {c['title']} ---\nContent:\n{c['text'][:per]}\n\n"
        sysp = ("You are a helpful assistant that answers questions about Fermilab research.\n"
                "Answer using ONLY the provided context. If the answer is not in the context, "
                "politely say you do not know. Do not invent facts.")
        user = f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    else:
        sysp = ("You are a helpful assistant that answers questions about Fermilab research. "
                "Answer the question directly and concisely.")
        user = question
    return [{"role": "system", "content": sysp}, {"role": "user", "content": user}]


def generate(model, tok, messages):
    import torch
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


# --------------------------------------------------------------- LLM judge
JUDGE_PROMPT = """You are grading an assistant's answer about Fermilab.

Question: {q}
Reference answer: {ref}
Assistant answer: {ans}

Rate the assistant answer on a 1-5 scale for CORRECTNESS (does it match the
reference facts?). Reply with ONLY a JSON object: {{"score": <1-5>, "why": "<short>"}}"""


def llm_judge(client, question, reference, answer):
    """Tra ve diem 1-5 tu Gemini. Loi -> None (bo qua cau do)."""
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=JUDGE_PROMPT.format(q=question, ref=reference, ans=answer),
        )
        txt = resp.text.strip().replace("```json", "").replace("```", "")
        return int(json.loads(txt)["score"])
    except Exception as e:
        print(f"  [judge skip] {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(RETR_DIR / "corpus_st7.jsonl"))
    ap.add_argument("--model", default=str(RETR_DIR / "ft-qwen3-fermilab"))
    ap.add_argument("--gold", default=str(REPO_ROOT / "eval" / "gold_qa.jsonl"))
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--split", default=None)
    ap.add_argument("--judge", action="store_true", help="bat LLM-as-judge (can GEMINI_API_KEY)")
    args = ap.parse_args()

    from retriever import Retriever

    gold = read_jsonl(args.gold)
    print(f"Loading retriever ({args.corpus}) and model ({args.model}) ...")
    ret = Retriever(args.corpus)
    model, tok = load_model(args.model)

    judge_client = None
    if args.judge:
        from google import genai
        judge_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    rows = []
    for q in gold:
        chunks = ret.retrieve(q["question"], top_k=args.top_k, split=args.split)
        ans_rag = generate(model, tok, build_messages(q["question"], chunks))
        ans_norag = generate(model, tok, build_messages(q["question"], []))

        row = {
            "id": q["id"], "question": q["question"],
            "f1_rag": round(token_f1(ans_rag, q["reference_answer"]), 3),
            "f1_norag": round(token_f1(ans_norag, q["reference_answer"]), 3),
            "answer_rag": ans_rag, "answer_norag": ans_norag,
        }
        if judge_client:
            row["judge_rag"] = llm_judge(judge_client, q["question"], q["reference_answer"], ans_rag)
            row["judge_norag"] = llm_judge(judge_client, q["question"], q["reference_answer"], ans_norag)
        rows.append(row)
        print(f"  {q['id']}: F1 RAG={row['f1_rag']}  No-RAG={row['f1_norag']}")

    # ----- summary -----
    n = len(rows)
    avg = lambda key: round(sum(r[key] for r in rows) / n, 3) if n else 0.0
    print(f"\n=== GENERATION EVAL  (n={n}) ===")
    print(f"  Avg token-F1  RAG   : {avg('f1_rag')}")
    print(f"  Avg token-F1  No-RAG: {avg('f1_norag')}")
    if judge_client:
        jr = [r["judge_rag"] for r in rows if r.get("judge_rag") is not None]
        jn = [r["judge_norag"] for r in rows if r.get("judge_norag") is not None]
        if jr:
            print(f"  Avg LLM-judge RAG   : {sum(jr)/len(jr):.2f} / 5  (n={len(jr)})")
        if jn:
            print(f"  Avg LLM-judge No-RAG: {sum(jn)/len(jn):.2f} / 5  (n={len(jn)})")

    out = REPO_ROOT / "eval" / "results_generation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": n, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nChi tiet luu tai: {out}")


if __name__ == "__main__":
    main()
