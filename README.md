# Email Spam Detection using NLP

An applied AI system that detects spam / phishing emails on the
[**SetFit/enron_spam**](https://huggingface.co/datasets/SetFit/enron_spam)
corpus (33,716 labelled ham / spam emails) by comparing classical machine
learning against a fine-tuned transformer.

The complete experiment lives in **`spam_detection_NLP.ipynb`** and a frozen
HTML render of an executed run is provided in
**`spam_detection_NLP.html`**.

---

## 1. Models compared

| # | Model                       | Feature representation       | Family        |
|---|-----------------------------|------------------------------|---------------|
| 1 | Support Vector Machine      | TF-IDF (unigram + bigram)    | Classical ML  |
| 2 | Random Forest               | TF-IDF (unigram + bigram)    | Classical ML  |
| 3 | DistilBERT (fine-tuned)     | Sub-word embeddings          | Transformer   |

This satisfies the brief's "at least two AI techniques" requirement
(classical ML model comparison **and** NLP + transformers).

### Headline results (20% held-out test set, SEED = 42)

| Model                   | Accuracy | Precision | Recall | F1     | AUC    |
|-------------------------|---------:|----------:|-------:|-------:|-------:|
| SVM (TF-IDF)            | 0.9913   | 0.9910    | 0.9918 | 0.9914 | 0.9994 |
| DistilBERT              | 0.9867   | 0.9933    | 0.9802 | 0.9867 | 0.9986 |
| Random Forest (TF-IDF)  | 0.9712   | 0.9538    | 0.9916 | 0.9723 | 0.9972 |

Full per-class reports, confusion matrices and ROC curves are reproduced in
`examples/sample_outputs.txt` and rendered in `spam_detection_NLP.html`.

---

## 2. Repository layout

```
Email Spam Detection Using NLP/
├── README.md                       # this file
├── requirements.txt                # Python dependencies
├── .gitignore
├── spam_detection_NLP.ipynb        # main notebook (the working prototype)
├── spam_detection_NLP.html         # static render of an executed run
└── examples/
    ├── sample_emails.txt           # six demo emails (3 ham + 3 spam)
    └── sample_outputs.txt          # metrics + predictions from a real run
```

---

## 3. Environment setup

The project targets **Python 3.10 or 3.11**.

### 3.1 Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users:** `requirements.txt` installs the default CPU build of
> PyTorch. If you have an NVIDIA GPU and want CUDA acceleration for
> DistilBERT fine-tuning, replace the torch install with the matching
> wheel from <https://pytorch.org/get-started/locally/>, e.g.
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 3.3 NLTK data packages

The notebook downloads these on first run (cell 2) but you can also fetch
them manually:

```bash
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['stopwords','wordnet','omw-1.4','punkt','punkt_tab']]"
```

### 3.4 Dataset

`SetFit/enron_spam` is downloaded automatically by `datasets.load_dataset`
on first run (cell 4). Internet access is required the first time; after
that it is cached under `~/.cache/huggingface/`.

---

## 4. Running the notebook

```bash
jupyter notebook spam_detection_NLP.ipynb
```

Run the cells top-to-bottom. The notebook is organised as:

1. Imports
2. Load dataset (SetFit/enron_spam)
3. Exploratory Data Analysis (7 visualisations)
4. Text-cleaning pipeline (lowercase / HTML / URLs / non-alpha / stopwords / lemmatisation)
5. Train / test split (stratified, 80 / 20)
6. TF-IDF feature extraction (unigram + bigram, 20k features)
7. Classical models - **SVM** and **Random Forest**
8. **DistilBERT** fine-tuning (2 epochs, batch 32, lr 2e-5, max_len 128)
9. Evaluation - metric table, bar chart, confusion matrices, ROC overlay
10. `predict_message(text)` - the deployable artefact
11. Interactive REPL

### 4.1 Expected runtime

| Stage                     | GPU (e.g. Colab T4) | CPU only          |
|---------------------------|--------------------:|------------------:|
| EDA + text cleaning       | 1 – 2 min           | 1 – 2 min         |
| TF-IDF + SVM + RF         | 1 – 3 min           | 2 – 5 min         |
| DistilBERT (2 epochs)     | ~1 min              | ~30 – 60 min      |
| **Total end-to-end**      | **~5 min**          | **~35 – 70 min**  |

DistilBERT trains on a sub-sample of 6,000 train / 1,500 test by default
(set via `TRAIN_SUBSAMPLE` / `TEST_SUBSAMPLE` in section 8) so it runs in
reasonable time on free-tier hardware. Comment out those lines in cell
*"MODEL_NAME = ..."* to fine-tune on the full split.

---

## 5. Using `predict_message()`

After running the notebook end-to-end, the function `predict_message(text)`
is available in memory. It auto-selects the strongest backend trained in
this kernel session:

```
DistilBERT  >  SVM (TF-IDF)  >  Random Forest (TF-IDF)
```

```python
predict_message("Subject: WIN A FREE iPHONE!! Click http://bit.ly/free-iphone")
# {'input': 'Subject: WIN A FREE iPHONE!! Click ...',
#  'label': 'spam',
#  'probability': 0.994,
#  'model': 'distilbert'}

predict_message("Subject: Lunch tomorrow\n\nWant to grab lunch around 1pm?")
# {'input': 'Subject: Lunch tomorrow ...',
#  'label': 'ham',
#  'probability': 0.028,
#  'model': 'distilbert'}
```

Force a specific backend with `predict_message(text, backend="svm")`,
`backend="rf"`, or `backend="distilbert"`.

For a typed-input loop, run the final cell (Section 11 - Interactive REPL)
and submit `quit` to exit.

See `examples/sample_emails.txt` for input messages and
`examples/sample_outputs.txt` for the corresponding expected outputs.

---

## 6. Reproducibility

- `SEED = 42` is set for `random`, `numpy`, the train/test split, and
  Hugging Face `TrainingArguments(seed=...)`.
- The DistilBERT pipeline reuses the exact same train / test row indices
  as the classical pipeline, so all three models are evaluated on the same
  split (no leakage and a fair comparison).
- Re-running the notebook should reproduce the headline metrics within
  about 1% (CUDA non-determinism aside).

---

## 7. Notes / limitations

- DistilBERT is fine-tuned on a sub-sample (6k train / 1.5k test) to keep
  runtime reasonable on free-tier hardware - this is flagged explicitly
  in section 8 of the notebook.
- Trained model artefacts are **not** persisted to disk by default; they
  live in the kernel session. Restart-and-run-all to use
  `predict_message()` in a new session.
- The text-cleaning pipeline (lowercase, HTML / URL strip, non-alpha
  removal, stopwords, lemmatisation) is applied to the classical models
  only. DistilBERT receives raw `subject + text`, by design, because
  transformers handle context directly.

---

## 8. Dataset citation

> SetFit / Enron Spam dataset, hosted on Hugging Face:
> <https://huggingface.co/datasets/SetFit/enron_spam>
