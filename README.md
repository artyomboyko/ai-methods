# AI Methods

<p align="center">
  <strong>Hands-on experiments in fine-tuning, parameter-efficient adaptation, speech recognition, and LLM quantization.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000" alt="Hugging Face">
  <img src="https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-6F42C1" alt="PEFT">
  <img src="https://img.shields.io/badge/Quantization-FP8%20%7C%20NVFP4%20%7C%20INT4-0969DA" alt="Quantization">
</p>

This repository is a practical research and learning workspace for testing modern AI methods in reproducible notebooks. The current experiments focus on **LLM fine-tuning**, **PEFT / LoRA / QLoRA**, **Whisper fine-tuning for Russian ASR**, and **post-training quantization**.

Rather than collecting isolated demos, the notebooks are structured around explicit baselines, controlled comparisons, and measurable evaluation.

## Current focus

- **Parameter-Efficient Fine-Tuning** — Prompt Tuning, LoRA, and QLoRA with frozen base models and compact trainable adapters.
- **LLM evaluation** — generation-based and forced-choice evaluation, Accuracy, Macro F1, class-level Precision / Recall / F1, valid-output checks, and confusion matrices.
- **Speech recognition** — fine-tuning `openai/whisper-small` for Russian ASR with a modern Hugging Face audio pipeline.
- **Quantization** — controlled comparison of BF16, FP8, NVFP4, and INT4 / GPTQ variants using LLM Compressor.
- **Reproducible environments** — a VS Code Dev Container setup with pinned core dependencies for the LLM experiments.

## Notebook catalog

| Area | Notebook | What it explores |
|---|---|---|
| PEFT | [`1_peft-basics-results-table-fixed.ipynb`](notebooks/finetuning/peft/1_peft-basics-results-table-fixed.ipynb) | Prompt Tuning on `Qwen/Qwen3.5-2B-Base` for SST-2 generative sentiment classification. The base model remains frozen while virtual prompt embeddings are trained. |
| LoRA | [`2_lora-torchmetrics-results-table-fixed.ipynb`](notebooks/finetuning/peft/2_lora-torchmetrics-results-table-fixed.ipynb) | LoRA on the same Qwen3.5-2B base model and SST-2 task, enabling a direct comparison with Prompt Tuning under closely matched conditions. |
| QLoRA | [`4_qlora.ipynb`](notebooks/finetuning/peft/4_qlora.ipynb) | QLoRA on `Qwen/Qwen3.5-2B-Base` with 4-bit NF4, double quantization, BF16 compute, and LoRA adapters. |
| ASR | [`Whisper_fine_tune_2026_formatted.ipynb`](notebooks/finetuning/asr/Whisper_fine_tune_2026_formatted.ipynb) | Updated 2026 pipeline for fine-tuning `openai/whisper-small` on Russian speech using current Transformers / Datasets audio APIs. |
| ASR archive | [`Whisper_fine_tune_old.ipynb`](notebooks/finetuning/asr/Whisper_fine_tune_old.ipynb) | Earlier Whisper fine-tuning implementation kept for comparison with the modernized pipeline. |
| Quantization | [`quantization.ipynb`](notebooks/quantization/quantization.ipynb) | Post-training quantization study comparing BF16 baseline, FP8, NVFP4, and INT4 W4A16 + GPTQ with a shared evaluation protocol. |

## Fine-tuning experiments

### Prompt Tuning

The PEFT baseline uses `Qwen/Qwen3.5-2B-Base` on the Stanford SST-2 sentiment task. Classification is formulated through causal generation with the verbalizers `positive` and `negative`.

Only virtual prompt embeddings are trainable; the base model remains frozen. The notebook evaluates the model with conditional log-likelihood as well as free generation, making it possible to separate **task understanding** from **output-format compliance**.

### LoRA

The LoRA notebook deliberately reuses the same base model, dataset, prompt format, and validation setup as the Prompt Tuning experiment. This makes the comparison more meaningful than evaluating unrelated methods on different conditions.

The experiment studies low-rank adaptation while keeping pretrained weights frozen and training only compact LoRA matrices.

### QLoRA

The QLoRA experiment combines LoRA with a 4-bit NF4 base model loaded through bitsandbytes. It uses:

- NF4 4-bit weight quantization;
- double quantization;
- BF16 compute;
- `prepare_model_for_kbit_training()`;
- LoRA across linear layers;
- a paged AdamW optimizer.

Evaluation is performed before and after adaptation with the same family of classification and output-quality metrics.

## Speech recognition

The current ASR notebook fine-tunes `openai/whisper-small` for Russian speech recognition. The 2026 version modernizes the older pipeline for current Hugging Face tooling:

- dynamic audio decoding and resampling through the modern Datasets audio stack;
- feature extraction inside the data collator instead of materializing a large intermediate feature dataset;
- `AutoProcessor` and `AutoModelForSpeechSeq2Seq`;
- current `Seq2SeqTrainer` APIs;
- BF16 / TF32 on compatible NVIDIA GPUs;
- WER-oriented ASR evaluation.

The older Whisper notebook is retained as a historical reference so the evolution of the pipeline remains visible.

## Quantization study

The quantization notebook compares several representations of the same model family under a shared evaluation setup:

| Variant | Approach |
|---|---|
| **BF16** | Baseline floating-point model; used as the reference rather than treated as quantization. |
| **FP8** | W8A8-style FP8 quantization with dynamically quantized activations. |
| **NVFP4** | 4-bit floating-point weights and activations with Blackwell-oriented scaling. |
| **INT4** | W4A16 integer weight quantization with GPTQ and calibration data. |

Each quantization experiment starts from a fresh copy of the same BF16 base model rather than chaining one quantization method on top of another. The variants are then evaluated on the same SST-2 validation subset with accuracy, Macro F1, class metrics, valid-output rate, confusion matrices, and generation timing.

## Evaluation philosophy

The notebooks are built around a simple rule: **change one method, keep the comparison conditions as stable as possible**.

That means using matched datasets and prompts when comparing PEFT methods, measuring models both before and after adaptation, separating free-generation quality from forced-choice task accuracy, and reloading the original BF16 model before every quantization experiment.

The goal is not just to make a model run, but to make the effect of the method visible and measurable.

## Core stack

`Python` · `PyTorch` · `Transformers` · `Datasets` · `PEFT` · `TRL` · `bitsandbytes` · `TorchMetrics` · `LLM Compressor` · `Plotly` · `Accelerate` · `jiwer`

For the main LLM experiments, the repository also includes a `.devcontainer` setup with Docker / VS Code configuration and pinned core dependencies.

## Repository structure

```text
ai-methods/
├── .devcontainer/                 # reproducible development environment
├── notebooks/
│   ├── finetuning/
│   │   ├── asr/                   # Whisper fine-tuning
│   │   └── peft/                  # Prompt Tuning, LoRA, QLoRA
│   └── quantization/              # BF16 / FP8 / NVFP4 / INT4 study
├── LICENSE
└── README.md
```

## Status

This is an evolving experimental repository. New notebooks are added when a method is worth testing in a reproducible, measurable setup rather than as a standalone code snippet.

## License

This repository is licensed under the **GNU General Public License v3.0**.
