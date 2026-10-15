# HealthAssist AI

HealthAssist AI is an educational medical question-answering prototype developed as a machine learning capstone project.

The system retrieves information from the MedQuAD dataset and displays source information when a suitable answer is available. It is not a medical diagnostic tool.

## Project Objectives

- Explore machine learning techniques for medical question answering.
- Compare retrieval-based and generative NLP approaches.
- Develop an interactive prototype.
- Investigate the limitations of medical question-answering systems.

## Dataset

The project uses the MedQuAD dataset (`lavita/MedQuAD`).

After preprocessing, the dataset contained approximately 16,357 question-answer pairs.

The dataset was divided into training, validation, and test sets using question-focus grouping.

| Split | Records |
|---|---:|
| Training | 13,045 |
| Validation | 1,674 |
| Test | 1,638 |

## Machine Learning Approaches

### TF-IDF Retrieval

A TF-IDF vectorizer represents medical questions as numerical features.

Cosine similarity is used to identify related questions in the training dataset.

### Fine-Tuned T5

The `google-t5/t5-small` model was fine-tuned for medical question answering and compared against the TF-IDF baseline.

### HealthAssist AI v6

The interactive prototype uses a retrieval-grounded approach with:

- Medical condition extraction
- Question intent detection
- Condition and intent matching
- TF-IDF similarity ranking
- Answer quality filtering
- Source attribution
- Limited-information responses
- Preliminary rule-based safety checks

## Evaluation Results

Held-out test results from the original model comparison:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| TF-IDF baseline | 0.3545 | 0.1798 | 0.2581 |
| Fine-tuned T5-small | 0.3697 | 0.2534 | 0.3286 |

These ROUGE scores measure textual overlap, not medical correctness.

In an exploratory set of 20 selected questions, HealthAssist AI v6 returned:

- 14 retrieved answers
- 1 limited-information response
- 5 unanswered questions

These exploratory results are not an independent clinical evaluation.

## Application Features

- Gradio user interface
- Example medical questions
- Educational answers from available sources
- Source display
- Explicit unsupported-question messaging
- Medical disclaimer

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

The application loads its retrieval components from the same folder.

## Project Files

- `app.py` — Gradio application
- `healthassist_v6.py` — retrieval and safety functions
- `tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer
- `train_tfidf_matrix.joblib` — training question vectors
- `train_conditions.joblib` — extracted condition labels
- `train_intents.joblib` — detected intent labels
- `retrieval_data.csv` — question-answer-source data used by the application
- `requirements.txt` — Python dependencies


## Dataset Availability

The application requires `retrieval_data.csv`, containing the
13,045 training records used to build the saved TF-IDF matrix.

This file is not currently included in the public repository
because redistribution permissions have not yet been verified.

The original dataset is available from:
https://huggingface.co/datasets/lavita/MedQuAD

Downloading MedQuAD alone does not reproduce the exact training
split or row order required by this application's saved matrix.
A reproducible data preparation workflow is still needed.

## Limitations

- The dataset may contain incomplete or outdated medical information.
- The system cannot reliably diagnose symptoms or assess emergencies.
- Some questions have no suitable answer in the available dataset.
- Rule-based safety checks may miss unsafe or ambiguous requests.
- Retrieval quality and clinical correctness have not been independently validated.
- Model performance scores do not establish medical reliability.

## Medical Disclaimer

HealthAssist AI is an educational research prototype. It is not intended for clinical use or medical decision-making. Always consult a qualified healthcare professional for medical concerns. For emergencies, seek urgent medical care.

## Data Source

MedQuAD: https://huggingface.co/datasets/lavita/MedQuAD

## Reconstructing the Retrieval Dataset

The retrieval dataset is not included directly in this repository.

HealthAssist AI v6 uses 13,045 training question-answer records from the
Hugging Face `lavita/MedQuAD` dataset.

The repository includes:

- `training_manifest.json` — ordered SHA-256 fingerprints identifying the training records.
- `reconstruct_dataset.py` — script that reconstructs the retrieval dataset.

### Reconstruction steps

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Reconstruct the dataset:

```bash
python reconstruct_dataset.py
```

The script downloads MedQuAD, identifies the selected records, preserves their
original training order, verifies their fingerprints, and creates
`retrieval_data.csv` in the project directory.

### Verification

The reconstruction was tested against the original HealthAssist AI v6 retrieval
dataset:

- Records reconstructed: **13,045**
- Fingerprint verification: **PASS**
- Row order verification: **PASS**
- Dataset equality check: **PASS**

Reconstruction depends on the source dataset remaining available and compatible.

The training manifest identifies the exact retrieval records; it does not
independently reproduce the original train/validation/test splitting procedure.
