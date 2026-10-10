# Inbox Shield

A local email-classification prototype with a dashboard, a FastAPI backend, and a scikit-learn text model. It distinguishes legitimate (ham) email, unsolicited spam, and phishing attempts; shows class confidence and influential terms; stores scan summaries in SQLite; and can retrain from a labeled CSV.

## Run on Windows

From the project root in PowerShell, with Python 3.11 or newer installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The API documentation is at `http://127.0.0.1:8000/docs`. Keep the server running while using the dashboard; press `Ctrl+C` in the terminal to stop it.

Run the backend tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

## Dashboard

- **Overview:** scan totals, ham/spam/phishing distribution, seven-day activity, and recent results.
- **Analyze email:** classify a message and inspect per-class confidence and the terms that most influenced the result.
- **Scan history:** search and filter saved classification summaries.
- **Model & data:** review accuracy, spam precision/recall and false-positive rate, per-class scores, and the three-class confusion matrix; upload a dataset and select Naive Bayes, Logistic Regression, or Linear SVM.

The app is served from the same origin as the API. Email text is classified locally and is not sent to a third-party service. History stores the sender, a short preview, classification, confidence, and keyword signals; full message bodies are not retained. The SQLite database is created at `backend/spam_history.db`.

## Training data

The included `backend/data/spam.csv` is a small, hand-written demo set so the app works immediately. Its evaluation numbers are not a measure of real-world performance and should not be used to make production filtering decisions.

On the **Model & data** page, upload a CSV with a label column and a message column. Supported header pairs include `label,text`, `category,message`, `class,content`, and `v1,v2`. Labels are `ham`, `spam`, and `phishing`, or numeric `0`, `1`, and `2` respectively. Headerless two-column files are also accepted. Include at least 12 valid rows with at least two examples from each class. Files are limited to 20 MB. Binary corpora such as the UCI SMS Spam Collection need a labeled phishing class added before they can train this three-class model.

The UCI SMS Spam Collection is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). It contains SMS messages rather than full emails and only ham/spam labels, so it is useful for experiments after phishing examples are added, but is not a substitute for a representative email corpus. The project brief also suggests SpamAssassin data; preprocess it into labeled ham/spam/phishing rows before upload. Retraining uses a fixed, stratified 75/25 train/test split. Predictions use the highest class probability; Linear SVM scores are softmax-scaled for display and are not calibrated probabilities.

## API

- `GET /api/health` — service and active model status.
- `GET /api/dashboard` — counts, seven-day series, recent scans, and model metrics.
- `POST /api/classify` — JSON body `{ "text": "...", "sender": "optional@example.com" }`.
- `GET /api/history?limit=50&q=term&label=all|ham|spam|phishing` — saved scan summaries.
- `GET /api/model` — evaluation metrics and confusion matrix.
- `POST /api/dataset` — multipart fields `file` and `model` for dataset upload and retraining.

This is a local prototype, not a production email gateway. It has no authentication, multi-user isolation, calibrated risk guarantees, or production-grade abuse controls. Use a representative, properly licensed dataset and evaluate false positives before deployment.