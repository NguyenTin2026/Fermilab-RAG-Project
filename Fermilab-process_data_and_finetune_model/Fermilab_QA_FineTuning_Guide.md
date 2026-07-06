# Fermilab QA Fine-Tuning Guide

This document describes how to fine-tune and validate the Fermilab Q&A assistant model in `Fermilab-process_data_and_finetune_model`.

## 1. Purpose

This guide is the single place to find:
* the recommended environment and dependencies for QLoRA training,
* the exact training command for `03_finetune_qlora.py`,
* validation chat and merge commands,
* notes for running the Streamlit app with the fine-tuned model.

## 2. Prerequisites

From the repository root:
```bash
cd /home/st7/Fermilab-Project/Fermilab-process_data_and_finetune_model
source ndo1/bin/activate
```

Install required Python packages if needed:
```bash
python3 -m pip install --break-system-packages torch transformers trl peft bitsandbytes accelerate datasets streamlit
```

Confirm you have a CUDA GPU available and enough VRAM (12GB+ recommended).

## 3. Fine-Tuning Command

Train a QLoRA adapter using the synthetic `train_qa.jsonl` dataset:
```bash
python3 03_finetune_qlora.py --data train_qa.jsonl --out ./ft-qwen3-fermilab
```

This creates the output adapter directory `./ft-qwen3-fermilab`.

### Notes
* `03_finetune_qlora.py` is already configured for 4-bit NF4 quantization.
* The script uses gradient accumulation to keep the effective batch size larger while staying within GPU memory.
* If training fails due to OOM, reduce `per_device_train_batch_size` or lower `max_length` in the script.

## 4. Validation Chat

Run a quick before/after comparison between the base model and the fine-tuned adapter:
```bash
python3 03_finetune_qlora.py --chat ./ft-qwen3-fermilab
```

This loads the base `Qwen/Qwen3-4B-Instruct-2507` model and then applies the adapter for direct comparison.

## 5. Merge Adapter into Full Model

When you are satisfied with the adapter, bake it into full model weights for easier deployment:
```bash
python3 03_finetune_qlora.py --merge ./ft-qwen3-fermilab
```

The merged model is saved to `./ft-qwen3-fermilab/merged`.

## 6. Run the Streamlit App

Launch the Fermilab Q&A dashboard and ensure it loads the fine-tuned model:
```bash
streamlit run app.py --server.fileWatcherType none
```

If the app does not find a merged model, it will attempt to load the adapter directly and fall back to the base model.

## 7. Helpful Reminders

* Keep `./ft-qwen3-fermilab` in the `Fermilab-process_data_and_finetune_model` folder.
* If you want the app to show the fine-tuned model status, generate or merge the model before starting `app.py`.
* Use the repository `README.md` for broader project context and this guide for the specific fine-tuning workflow.
