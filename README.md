# TF-IDF Based SMS Spam Detection System

A beginner-friendly machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using **TF-IDF** feature extraction and **Logistic Regression**.

## 1. Project structure

```text
TFIDF_SMS_Spam_Detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── README.txt
├── models/
│   └── (model and evaluation metrics are created after training)
└── screenshots/
    └── (save your actual output screenshots here)
```

## 2. Requirements

- Windows 10/11
- Python 3.10 or newer
- Internet connection for the first dataset download

## 3. Setup in Windows

1. Extract this ZIP file.
2. Open the extracted folder.
3. Click the folder address bar, type `cmd`, and press Enter.
4. Create a virtual environment:

```bash
python -m venv .venv
```

5. Activate it:

```bash
.venv\Scripts\activate
```

6. Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Train the model

Run:

```bash
python train_model.py
```

The program downloads the UCI SMS Spam Collection dataset into `data/` the first time it runs, trains the model, prints evaluation results, and saves the trained model and metrics in `models/`.

## 5. Run the web application

Run:

```bash
streamlit run app.py
```

A browser tab should open. Enter a message and click **Check Message**.

## 6. Required algorithm: TF-IDF

TF-IDF means Term Frequency–Inverse Document Frequency. It converts text into numerical values.

- **Term Frequency (TF):** how often a term appears in a message.
- **Inverse Document Frequency (IDF):** reduces the weight of terms that appear in many documents.
- **TF-IDF:** combines both values to give a weight to each term.

This project uses `TfidfVectorizer` from scikit-learn inside a machine-learning Pipeline. The TF-IDF vectorizer is fitted only on the training messages to avoid test-data leakage.

## 7. Model

Logistic Regression is used as the classification algorithm. The data is split into 80% training and 20% testing, using stratification and a fixed random seed for reproducibility.

Evaluation includes accuracy, precision, recall, F1-score, and a confusion matrix. Use the values printed by your own run in your report; do not assume a particular score in advance.

## 8. Dataset and attribution

Dataset: **SMS Spam Collection** by Tiago Almeida and José Hidalgo (2011), UCI Machine Learning Repository.

- Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- DOI: https://doi.org/10.24432/C5CC84
- License identified by UCI: Creative Commons Attribution 4.0 International (CC BY 4.0).

The full dataset is downloaded by `train_model.py` when you first train the model. An internet connection is required for this first download.

## 9. Screenshots for submission

After running the app, take genuine screenshots of:
1. The home screen.
2. A message predicted as SPAM.
3. A message predicted as NOT SPAM.
4. The training terminal showing the evaluation report.

Save them in the `screenshots/` folder. Do not submit example or fabricated performance values.

## 10. GitHub

Create a new public or private GitHub repository and upload the extracted project folder. Do not upload the `.venv/` folder. If you later create a GitHub repository, add its URL to your submission document.

## 11. Troubleshooting

- If `python` is not recognized, reinstall Python and select **Add Python to PATH**, or try `py` instead of `python`.
- If package installation fails, check your internet connection and run `python -m pip install -r requirements.txt`.
- If dataset download fails, check internet access and try `python train_model.py` again.
- If Streamlit is not recognized, run `python -m streamlit run app.py`.
