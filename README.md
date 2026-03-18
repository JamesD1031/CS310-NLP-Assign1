# CS310 NLP Assignment 1

This repository contains a submission-ready version of CS310 Natural Language Processing Assignment 1: neural text classification for Chinese humor detection.

## Project Layout

- `Assignment/A1/A1_nn.ipynb`: main notebook submission
- `Assignment/A1/A1_report.md`: report source
- `Assignment/A1/A1_written_solution.md`: written derivations
- `Assignment/A1/data_utils.py`: minimal local helper used by the notebook
- `Assignment/A1/train.jsonl`: training data
- `Assignment/A1/test.jsonl`: labeled held-out test data
- `Assignment/A1/requirements_a1.txt`: Python dependencies for reproduction

## Python Environment

Tested in a clean local virtual environment with Python `3.14`.

The notebook only relies on:

- `torch`
- `jieba`
- `numpy`

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Assignment/A1/requirements_a1.txt
```

## How To Run

You can run the notebook from either:

- the repository root, then open `Assignment/A1/A1_nn.ipynb`
- the notebook directory `Assignment/A1`

The notebook resolves its own data/helper paths, so it does not depend on hardcoded `Assignment/A1/...` or `Lab/Lab2/...` imports.

Typical workflow:

```bash
source .venv/bin/activate
jupyter notebook
```

Then open:

```text
Assignment/A1/A1_nn.ipynb
```

Any Jupyter frontend is fine as long as it uses the virtual environment above.

## Submission Files

The actual submission files are:

- `A1_nn.ipynb`
- `A1_report.md`
- `A1_written_solution.md`
- `requirements_a1.txt`
- `data_utils.py`

The dataset files are included in the repo for reproducibility. If your course submission only wants the notebook/report/written answers, export or package those separately as needed.

## Expected Runtime

On CPU, the notebook should finish in a few minutes or less. The model trains for 12 epochs with a bag-of-words `EmbeddingBag` architecture.

## Reproducibility Notes

- The notebook sets a fixed random seed (`310`).
- Labels from the JSONL files are converted from one-element lists to scalar integers.
- The notebook uses a stratified train/validation split from `train.jsonl`.
- The final evaluation is reported on the provided labeled `test.jsonl`.
- The notebook does not auto-install packages; missing dependencies should be installed from `requirements_a1.txt` before running.
