---
title: HealthAssist AI
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# HealthAssist AI

**An educational, English-language healthcare question-answering research prototype.**

[**Try the live Gradio application**](https://healthassist-ai-2.onrender.com/) · [**View the source code**](https://github.com/Tapiwanashe6/HealthAssist-AI)

Developed for the **Machine Learning Techniques I** summative project at African Leadership University. HealthAssist AI investigates how a traditional question-retrieval baseline compares with a fine-tuned Transformer for domain-specific medical question answering. The public application uses a retrieval-grounded pipeline that displays source information when available.

> **Medical disclaimer:** This is an educational research prototype, **not** a diagnostic, emergency-triage, or clinical decision-support system. Do not use its answers to make medical decisions. Consult a qualified healthcare professional for personal medical questions.

## Research question and objectives

**How do TF-IDF retrieval and a fine-tuned T5-small model compare on English-language medical question answering, and what limitations appear when deploying a constrained educational assistant?**

The project aims to:
- Prepare and investigate real-world medical question–answer data.
- Establish a TF-IDF retrieval baseline and fine-tune T5-small.
- Compare approaches using held-out ROUGE metrics and qualitative error analysis.
- Build a web interface for exploring available medical information.
- Document unsupported questions, source limitations, and safety constraints.

## African-context relevance

Access to understandable health information is a relevant challenge in African communities. An educational question-answering interface could potentially help users understand general medical terms and prepare questions for healthcare professionals. This project demonstrates a technical foundation that could later be adapted to local information needs.

**Important scope distinction:** MedQuAD is an English-language dataset drawn largely from United States medical-information sources. It is **not an African-specific dataset**, and HealthAssist AI has **not** been evaluated for local clinical guidance, African languages, or health outcomes. Future work would require locally reviewed questions, appropriate regional medical sources, and separate language and safety evaluations.

## Dataset and preparation

**Dataset:** [MedQuAD on Hugging Face](https://huggingface.co/datasets/lavita/MedQuAD) · [Original MedQuAD repository](https://github.com/abachaa/MedQuAD)

The initial dataset contained **47,441** records. Following cleaning and quality checks, **16,357** question–answer pairs were retained. Group-aware splitting was used to reduce obvious overlap among related questions.

| Split | Question–answer pairs | Use |
|---|---:|---|
| Training | 13,045 | Retrieval index and model fine-tuning |
| Validation | 1,674 | Configuration comparisons |
| Test | 1,638 | Final held-out evaluation |
| **Total cleaned** | **16,357** | |

The original corpus may include incomplete, duplicated, boilerplate-heavy, or outdated material. Group-aware splitting reduces obvious leakage but cannot guarantee that semantically similar medical facts do not appear across splits.

## Modeling approaches

### 1. TF-IDF retrieval baseline

Training questions are represented using term-frequency/inverse-document-frequency features. Cosine similarity ranks the stored questions against a new input, and the associated answer is retrieved. This offers a straightforward, interpretable baseline.

### 2. Fine-tuned T5-small (experimental model)

The pretrained `google-t5/t5-small` sequence-to-sequence Transformer was fine-tuned on medical question–answer pairs. It was evaluated against the TF-IDF baseline using held-out reference answers.

**The fine-tuned T5 model is a research comparison, not the model serving answers in the public Gradio app.** Higher text-overlap scores do not guarantee factual or clinical reliability.

### 3. HealthAssist AI v6 (deployed application)

The public app uses the saved TF-IDF retrieval components, plus heuristic controls:

1. Check selected potentially unsafe requests.
2. Identify medical-condition and question-intent cues.
3. Filter or rank available training questions using condition, intent, and TF-IDF similarity.
4. Apply answer-quality checks and display an answer with source information when available.
5. Return limited information or decline unsupported questions when the rules determine that an adequate answer is unavailable.

These rules are incomplete safeguards, **not clinically validated medical triage**.

## Experiments and results

### Final held-out test — 1,638 questions

| Approach | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---:|---:|---:|
| TF-IDF baseline | 0.3545 | 0.1798 | 0.2581 |
| Fine-tuned T5-small | **0.3697** | **0.2534** | **0.3286** |

T5-small produced greater reference-text overlap, particularly for ROUGE-2 and ROUGE-L. **ROUGE measures lexical similarity, not medical correctness, completeness, or safety.** These scores describe the evaluated research models, not clinical accuracy of the deployed v6 app.

### Exploratory application checks

In **20 selected questions** tested during development, the retrieval-grounded prototype produced:

| Response category | Count |
|---|---:|
| Retrieved answer | 14 |
| Limited-information response | 1 |
| No answer | 5 |

This small, selected set is a functional exploration, **not** an independent clinical validation or a representative accuracy estimate.

### Observed limitations

- Similar vocabulary can retrieve an answer about the wrong medical condition.
- Long reference answers may contain boilerplate or irrelevant details.
- Generated answers can repeat text or make unsupported claims.
- Some questions are outside the dataset's coverage.
- Simple rule-based restrictions can miss unsafe or ambiguous queries.

## Run the application locally

**Prerequisites:** Python, an internet connection for initial dataset reconstruction, and sufficient memory for the saved retrieval matrix and dataset.

```bash
git clone https://github.com/Tapiwanashe6/HealthAssist-AI.git
cd HealthAssist-AI
python -m pip install -r requirements.txt
python reconstruct_dataset.py
python app.py
```

Open the local URL printed by Gradio. The app can also attempt reconstruction automatically at startup if `retrieval_data.csv` is missing, but running the script explicitly makes the process easier to inspect.

### Why the retrieval CSV is reconstructed

`retrieval_data.csv` is not included in the public repository because redistribution permissions have not been confirmed. The repository includes:

- `training_manifest.json`: ordered SHA-256 fingerprints for the **13,045** training records.
- `reconstruct_dataset.py`: downloads MedQuAD, identifies the selected records, checks their fingerprints, and writes `retrieval_data.csv` in the required row order.

The reconstruction workflow was previously tested with **13,045 records**, passing fingerprint, order, and dataset-equality checks. It depends on the upstream MedQuAD distribution remaining available and compatible.

**Reproducibility boundary:** The manifest reconstructs the selected retrieval records; it does not independently recreate the full original train/validation/test splitting procedure or rerun T5 training. The public app loads saved retrieval artifacts and does not require a T5 checkpoint.

## Repository components

| File | Purpose |
|---|---|
| `app.py` | Gradio interface and startup logic |
| `healthassist_v6.py` | Retrieval and heuristic answer controls |
| `tfidf_vectorizer.joblib` | Saved fitted TF-IDF vectorizer |
| `train_tfidf_matrix.joblib` | Saved training-question vectors |
| `train_conditions.joblib` | Saved condition labels |
| `train_intents.joblib` | Saved question-intent labels |
| `training_manifest.json` | Ordered fingerprints for reconstruction |
| `reconstruct_dataset.py` | Dataset reconstruction and verification |
| `requirements.txt` | Pinned application dependencies |

The academic model-comparison results are documented in the research report and experimental notebook. This repository's deployment path specifically serves the retrieval-grounded prototype.

## Deployment

**Live app:** https://healthassist-ai-2.onrender.com/

The application is deployed with Gradio on Render. On a fresh instance, startup may require reconstructing the retrieval dataset; availability and startup time depend on hosting resources and the upstream dataset. The public app does not send questions to a generative medical chatbot API.

## Limitations and responsible use

- MedQuAD content may be outdated, incomplete, or unsuitable for a particular country.
- The dataset does not establish African clinical or language coverage.
- Retrieved answers may be irrelevant despite sharing vocabulary with a question.
- Neither ROUGE scores nor exploratory functional tests establish medical safety.
- Heuristic request filtering cannot reliably detect every emergency or unsafe medical question.
- Dataset access and reuse remain subject to upstream availability and applicable licenses.

## Sources and acknowledgments

- [MedQuAD dataset](https://github.com/abachaa/MedQuAD) — A. Ben Abacha and D. Demner-Fushman, *A Question-Entailment Approach to Question Answering*, BMC Bioinformatics 20, article 511 (2019), [DOI: 10.1186/s12859-019-3119-4](https://doi.org/10.1186/s12859-019-3119-4).
- [Hugging Face MedQuAD distribution](https://huggingface.co/datasets/lavita/MedQuAD)
- [T5-small pretrained model](https://huggingface.co/google-t5/t5-small)
- [scikit-learn](https://scikit-learn.org/stable/) — TF-IDF and baseline retrieval tools.
- [Gradio](https://www.gradio.app/) — Interactive web interface.

**Author:** Tapiwanashe Gift Marufu · African Leadership University

*Educational use only. Seek qualified medical care for health concerns.*
