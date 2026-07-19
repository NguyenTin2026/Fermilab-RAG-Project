# Evaluation harness

Đo chất lượng **thật** của hệ thống Q&A — thứ mà loss/token-accuracy lúc train không nói lên được. Có hai script, chạy trên máy có GPU + weights + corpus (`git lfs pull` trước).

## 1. Retrieval quality — `eval_retrieval.py`
Đo xem retriever có kéo đúng tài liệu liên quan lên top không.

```bash
python eval/eval_retrieval.py \
    --corpus Fermilab-process_data_and_finetune_model/corpus_st7.jsonl \
    --gold   eval/gold_qa.jsonl --k 1,3,5
```

In ra `Recall@1/3/5` và `MRR@k`. Relevance hiện chấm bằng *keyword proxy* (một chunk được coi là liên quan nếu chứa từ khóa của câu hỏi trong `gold_qa.jsonl`). Muốn nghiêm túc hơn, thay bằng qrels gán tay (câu hỏi → id tài liệu đúng) và sửa hàm `is_relevant`.

## 2. Answer quality — `eval_generation.py`
Chạy **RAG vs No-RAG** trên tập gold, chấm điểm để trả lời trực tiếp "độ chính xác cao không".

```bash
export GEMINI_API_KEY=...        # chỉ cần nếu bật --judge
python eval/eval_generation.py \
    --corpus Fermilab-process_data_and_finetune_model/corpus_st7.jsonl \
    --model  Fermilab-process_data_and_finetune_model/ft-qwen3-fermilab \
    --gold   eval/gold_qa.jsonl --top_k 3 --judge
```

Chấm 2 cách: **token-F1** (so với đáp án tham chiếu, không cần API) và **LLM-as-judge** (Gemini chấm correctness 1–5, cần `--judge`). Kết quả chi tiết ghi ra `eval/results_generation.json`.

## Gold set — `gold_qa.jsonl`
12 câu hỏi Fermilab seed (NOvA, DUNE/LBNF, muon g-2, PIP-II, SeaQuest, Tevatron...). **Đây là seed để bạn mở rộng** — nên nâng lên 50–100 câu, lý tưởng là rút từ tập test 1.747 chunk đã tách riêng, để con số đáng tin hơn. Mỗi dòng: `question`, `reference_answer`, `keywords`.
