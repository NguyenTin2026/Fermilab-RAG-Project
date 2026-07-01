# Discovery Hub fine-tuning pipeline
# Run `make help` to see steps. Override variables on the command line, e.g.
#   make pairs DATA=../discovery_data_download MODEL=Qwen/Qwen3-Embedding-0.6B

DATA       ?= data
MODEL      ?= Qwen/Qwen3-Embedding-0.6B
OUT        ?= runs/qwen06b-ft
DATASET    ?= dataset
NPER       ?= 2
BATCH      ?= 16
EPOCHS     ?= 3

.PHONY: help inspect pairs dataset baseline train eval all clean

help:
	@echo "Discovery Hub fine-tuning pipeline"
	@echo "  make inspect    - parse raw data, print a summary (no API, no GPU)"
	@echo "  make pairs      - generate synthetic query/doc pairs via Gemini"
	@echo "  make dataset    - mine hard negatives, build train/eval splits"
	@echo "  make baseline   - evaluate the BASE model (do this BEFORE training)"
	@echo "  make train      - fine-tune the small model on the triplets"
	@echo "  make eval       - evaluate the FINE-TUNED model"
	@echo "  make all        - pairs -> dataset -> baseline -> train -> eval"
	@echo ""
	@echo "Vars: DATA=$(DATA) MODEL=$(MODEL) OUT=$(OUT) NPER=$(NPER) BATCH=$(BATCH)"

inspect:
	python -m discovery_ft.parse $(DATA)

pairs:
	python -m discovery_ft.generate_queries --in $(DATA) --out pairs.jsonl --n-per-doc $(NPER)

dataset:
	python -m discovery_ft.build_dataset --pairs pairs.jsonl --in $(DATA) \
		--base-model $(MODEL) --out-dir $(DATASET)

baseline:
	python -m discovery_ft.evaluate --corpus $(DATASET)/corpus.jsonl \
		--qrels $(DATASET)/eval_qrels.jsonl --model $(MODEL) --tag base

train:
	python -m discovery_ft.train --train $(DATASET)/train.jsonl \
		--eval $(DATASET)/eval.jsonl --model $(MODEL) --out $(OUT) \
		--epochs $(EPOCHS) --batch $(BATCH)

eval:
	python -m discovery_ft.evaluate --corpus $(DATASET)/corpus.jsonl \
		--qrels $(DATASET)/eval_qrels.jsonl --model $(OUT)/final --tag finetuned

all: pairs dataset baseline train eval

clean:
	rm -rf $(DATASET) pairs.jsonl $(OUT)
