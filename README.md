# ResolveAI — Customer Complaint Classifier

An NLP application that suggests a financial product category from a written consumer complaint. Built with Python, scikit-learn, and Streamlit.

## Live Demo

[Open ResolveAI](https://customer-complaint-classifier-b7vkhmowuyycb9nmxrjabi.streamlit.app)

## Features

- Classify individual complaints into five financial product categories.
- Compare model scores across categories.
- Explore the evaluation dataset’s category distribution.
- Inspect per-category performance and a confusion matrix.
- Download the evaluation report.

## Supported Categories

- Credit Card
- Credit Reporting
- Debt Collection
- Mortgages and Loans
- Retail Banking

## Dataset and Preparation

Source: [Consumer Complaints Dataset for NLP — Kaggle](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)

The downloaded dataset contained 162,421 rows. Preparation included:

- Removing missing or empty complaint text.
- Normalising case and whitespace.
- Removing complaints with conflicting category labels.
- Removing duplicate complaint text.

The cleaned dataset contains **124,300 complaints**.

Dataset files are excluded from this repository.

## Model

The classification pipeline combines:

- **TF-IDF:** unigram and bigram text features.
- **Logistic Regression:** balanced class weights.

The dataset was divided using an **80/20 stratified random split**, with random state 42. The TF-IDF vocabulary was learned from training data only.

## Evaluation Results

| Metric | Result |
|---|---:|
| Test accuracy | 85.13% |
| Macro F1 | 84.17% |
| Majority-class baseline accuracy | 45.19% |
| Training complaints | 99,440 |
| Test complaints | 24,860 |

### Per-Category Results

| Category | Precision | Recall | F1 Score | Test Complaints |
|---|---:|---:|---:|---:|
| Credit Card | 76.57% | 85.40% | 80.75% | 2,993 |
| Credit Reporting | 93.66% | 83.33% | 88.19% | 11,234 |
| Debt Collection | 77.54% | 83.00% | 80.17% | 4,200 |
| Mortgages and Loans | 79.91% | 88.90% | 84.17% | 3,740 |
| Retail Banking | 84.88% | 90.46% | 87.58% | 2,693 |

## Main Files

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `data_cleaning.py` | Dataset cleaning |
| `train_model.py` | Model training and evaluation |
| `models/complaint_classifier.joblib` | Saved classification pipeline |
| `reports/metrics.json` | Overall evaluation metrics |
| `reports/classification_report.csv` | Per-category results |
| `reports/confusion_matrix.csv` | Confusion matrix |
| `requirements.txt` | Python dependencies |

## Run Locally

Use Python 3.11 for the deployment-compatible environment.

Clone the repository:

```bash
git clone https://github.com/anuvarna-r/customer-complaint-classifier.git
cd customer-complaint-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Install dependencies and start the app using Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The saved model and reports allow the application to run without downloading the training dataset.

## Retrain the Model

Download and extract the source dataset, then place `complaints_processed.csv` inside the `data` folder.

Run:

```powershell
.\.venv\Scripts\python.exe data_cleaning.py
.\.venv\Scripts\python.exe train_model.py
```

Retraining updates the saved model and evaluation reports.

## Limitations

- The model supports English financial complaints within five categories.
- It can assign a category to unrelated text.
- Displayed category scores are model estimates, not verified confidence levels.
- Training text was already preprocessed; real user wording may differ.
- Similar complaint templates may remain after exact duplicate removal.
- Random-split evaluation does not establish performance on future complaints.
- The model does not determine whether a complaint is valid.
- The available processed dataset does not support analysis of trends over time.

Predictions should be reviewed before routing complaints. Use fictional examples without personal information in the public demo.

## Technology

Python · pandas · scikit-learn · Joblib · Plotly · Streamlit

## Author

**Anuvarna R.**  
Data Science & AI Portfolio# customer-complaint-classifier
An NLP application that classifies consumer financial complaints using machine learning and provides an interactive dashboard to analyse complaint trends and model performance.
