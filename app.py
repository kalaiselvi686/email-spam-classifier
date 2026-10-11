from __future__ import annotations

import io
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).parent / "data" / "spam.csv"
DB_PATH = Path(__file__).parent / "spam_history.db"
FRONTEND_PATH = ROOT / "frontend"
LABELS = {
    "ham": "ham", "legitimate": "ham", "not spam": "ham", "0": "ham",
    "spam": "spam", "1": "spam",
    "phishing": "phishing", "phish": "phishing", "2": "phishing",
}
CLASS_NAMES = ("ham", "spam", "phishing")
MODEL_OPTIONS = {"naive_bayes", "logistic_regression", "linear_svm"}


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=10, max_length=50000)
    sender: str = Field(default="", max_length=254)


def normalize_label(value: object) -> str | None:
    return LABELS.get(str(value).strip().lower())


def read_records(payload: bytes, filename: str = "dataset.csv") -> list[tuple[str, str]]:
    try:
        frame = pd.read_csv(io.BytesIO(payload), sep=None, engine="python")
        normalized = {str(column).strip().lower(): column for column in frame.columns}
        label_column = next((normalized[key] for key in ("label", "v1", "category", "class", "type") if key in normalized), None)
        text_column = next((normalized[key] for key in ("text", "message", "email", "content", "v2") if key in normalized), None)
        if label_column is None or text_column is None:
            frame = pd.read_csv(io.BytesIO(payload), sep=None, engine="python", header=None)
            if frame.shape[1] < 2:
                raise ValueError("Expected a label column and a text column.")
            label_column, text_column = frame.columns[0], frame.columns[-1]
        records = [(str(text).strip(), normalize_label(label)) for label, text in zip(frame[label_column], frame[text_column])]
        valid = [(text, label) for text, label in records if text and label is not None]
        counts = {label: sum(row_label == label for _, row_label in valid) for label in CLASS_NAMES}
        if len(valid) < 12 or any(counts[label] < 2 for label in CLASS_NAMES):
            raise ValueError("Include at least 12 labeled rows and 2 examples each for ham, spam, and phishing.")
        return valid
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not read {filename}: {error}") from error


class Classifier:
    def __init__(self, records: list[tuple[str, str]], dataset_name: str, model_name: str = "naive_bayes"):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", stop_words="english", ngram_range=(1, 2), min_df=1, max_features=30000)
        texts, labels = zip(*records)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )
        self.model = self._make_model(model_name)
        self.model.fit(self.vectorizer.fit_transform(train_texts), train_labels)
        test_features = self.vectorizer.transform(test_texts)
        predictions, _ = self._predict(test_features)
        matrix = confusion_matrix(test_labels, predictions, labels=CLASS_NAMES)
        spam_index = CLASS_NAMES.index("spam")
        non_spam_indices = [index for index, label in enumerate(CLASS_NAMES) if label != "spam"]
        non_spam_count = int(matrix[non_spam_indices, :].sum())
        true_spam = int(matrix[spam_index, spam_index])
        predicted_spam = int(matrix[:, spam_index].sum())
        self.metrics = {
            "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
            "precision": round(true_spam / predicted_spam, 4) if predicted_spam else 0,
            "false_positive_rate": round(float(matrix[non_spam_indices, spam_index].sum() / non_spam_count), 4) if non_spam_count else 0,
            "recall": round(float(recall_score(test_labels, predictions, labels=["spam"], average="macro", zero_division=0)), 4),
            "f1": round(float(f1_score(test_labels, predictions, labels=["spam"], average="macro", zero_division=0)), 4),
            "confusion_matrix": matrix.tolist(),
            "classes": list(CLASS_NAMES),
            "class_metrics": {
                label: {
                    "precision": round(float(precision_score(test_labels, predictions, labels=[label], average="macro", zero_division=0)), 4),
                    "recall": round(float(recall_score(test_labels, predictions, labels=[label], average="macro", zero_division=0)), 4),
                    "f1": round(float(f1_score(test_labels, predictions, labels=[label], average="macro", zero_division=0)), 4),
                }
                for label in CLASS_NAMES
            },
            "test_size": len(test_labels),
            "training_size": len(train_labels),
            "dataset_name": dataset_name,
            "is_demo": dataset_name == "Built-in demo sample",
            "model": model_name,
        }

    @staticmethod
    def _make_model(name: str):
        if name == "logistic_regression":
            return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        if name == "linear_svm":
            return LinearSVC(class_weight="balanced", random_state=42)
        return MultinomialNB(alpha=0.35)

    def _predict(self, features):
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)
        else:
            import numpy as np

            decision = self.model.decision_function(features)
            decision = np.atleast_2d(decision)
            shifted = decision - decision.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        predictions = self.model.classes_[probabilities.argmax(axis=1)]
        return predictions, probabilities

    def classify(self, text: str) -> dict:
        features = self.vectorizer.transform([text])
        predictions, probabilities = self._predict(features)
        label = str(predictions[0])
        class_probabilities = {str(class_name): float(probabilities[0, index]) for index, class_name in enumerate(self.model.classes_)}
        confidence = class_probabilities[label]
        runner_up = max((class_name for class_name in class_probabilities if class_name != label), key=class_probabilities.get)
        feature_names = self.vectorizer.get_feature_names_out()
        row = features.tocsr()
        weights = getattr(self.model, "feature_log_prob_", None)
        if weights is None:
            class_weights = self.model.coef_
        else:
            class_weights = weights
        label_index = list(self.model.classes_).index(label)
        runner_up_index = list(self.model.classes_).index(runner_up)
        contribution_weights = class_weights[label_index] - class_weights[runner_up_index]
        terms = []
        for index, value in zip(row.indices, row.data):
            contribution = float(value * contribution_weights[index])
            terms.append({"term": str(feature_names[index]), "direction": label if contribution >= 0 else runner_up, "weight": round(abs(contribution), 4)})
        terms.sort(key=lambda item: item["weight"], reverse=True)
        return {"label": label, "confidence": round(confidence, 4), "class_probabilities": {key: round(value, 4) for key, value in class_probabilities.items()}, "keywords": terms[:6]}


def initialize_database() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            sender TEXT NOT NULL,
            preview TEXT NOT NULL,
            text TEXT NOT NULL,
            keywords TEXT NOT NULL
        )""")


FALLBACK_CSV = """label,text
ham,Hi Alex are we still meeting for lunch today
ham,Please review the attached project notes before our call
ham,Thanks for sending the update I will take a look this afternoon
ham,Can you share the address for the office event
ham,Your appointment is confirmed for Tuesday at ten
ham,The team moved the planning meeting to room four
ham,I left the keys on your desk see you tomorrow
ham,Here are the slides from this morning presentation
ham,Let me know if you need anything else for the report
ham,We are having dinner with Sam and Priya this weekend
ham,The parcel was delivered to the reception desk today
ham,Could you call me when you arrive at the station
ham,The invoice is attached for the design work completed
ham,Thanks for helping with the launch yesterday
ham,Your library book is due next Friday please return it on time
ham,I will be working from home this afternoon
ham,The doctor asked me to confirm the follow up visit
ham,Happy birthday hope you have a lovely day
ham,Here is the recipe I mentioned at lunch
ham,The meeting notes are in the shared project folder
spam,Win a free prize now click the link to claim your reward
spam,Congratulations you have won a cash bonus claim now
spam,Limited time offer buy now and get a free gift
spam,You have been selected for an exclusive lottery prize
spam,Claim your guaranteed reward today call now
spam,Free entry in our weekly draw text WIN to 80080
spam,Get cheap loans approved today no credit check
spam,Act now your special offer expires tonight
spam,You won a free vacation click here to collect
spam,Claim your cash prize immediately reply with your details
spam,Lowest price pharmacy pills order now online
spam,Unlock an exclusive bonus offer today limited time
spam,Make money fast from home click to learn more
spam,Your account won a reward claim it now
spam,Free mobile offer call now to get started
spam,Special discount just for you buy today and save
spam,Winner selected claim your gift card here
spam,Get instant cash today apply now no paperwork
spam,Click here to receive your free voucher
spam,Final notice claim your prize before it expires
phishing,Your bank account is locked verify your password at the secure account link
phishing,We detected unusual sign in activity confirm your identity using this link
phishing,Your mailbox will be disabled unless you sign in and verify your account
phishing,Update your payroll details immediately using the employee portal link
phishing,Your payment failed review your billing information to restore access
phishing,Security alert reset your password now to prevent account suspension
phishing,The delivery address is incomplete confirm your details to release the package
phishing,Unusual activity detected sign in to your account to secure it
phishing,Your tax refund is waiting verify your identity and bank details here
phishing,Document shared with you sign in to view the confidential file
phishing,Your subscription payment failed update card details to avoid cancellation
phishing,We suspended your account click the verification link to restore access
phishing,Confirm your work email password to prevent mailbox closure
phishing,Review this urgent wire transfer request from the company director
phishing,Your account has a security hold verify credentials to continue
phishing,Sign in to review the attached invoice from your supplier
phishing,Confirm your identity now to release the pending bank transfer
phishing,Your package is on hold provide your address and card details
phishing,Shared document requires account verification before it can be viewed
"""


def get_records() -> list[tuple[str, str]]:
    try:
        return read_records(DATA_PATH.read_bytes(), DATA_PATH.name)
    except (OSError, ValueError):
        return read_records(FALLBACK_CSV.encode(), "built-in demo sample")


initialize_database()
classifier_lock = Lock()
classifier = Classifier(get_records(), "Built-in demo sample")

app = FastAPI(title="Inbox Shield", version="1.0.0")


def scan_dict(row: tuple) -> dict:
    return {"id": row[0], "created_at": row[1], "label": row[2], "confidence": row[3], "sender": row[4], "preview": row[5], "keywords": json.loads(row[7])}


@app.get("/api/health")
def health():
    return {"status": "ok", "model": classifier.model_name, "dataset": classifier.dataset_name}


@app.post("/api/classify")
def classify_email(request: ClassifyRequest):
    result = classifier.classify(request.text)
    created_at = datetime.now(timezone.utc).isoformat()
    preview = " ".join(request.text.split())[:120]
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            "INSERT INTO scans (created_at, label, confidence, sender, preview, text, keywords) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (created_at, result["label"], result["confidence"], request.sender.strip(), preview, "", json.dumps(result["keywords"])),
        )
        result["id"] = cursor.lastrowid
        result["created_at"] = created_at
    return result


@app.get("/api/history")
def history(limit: int = Query(default=50, ge=1, le=250), q: str = "", label: Literal["all", "spam", "ham", "phishing"] = "all"):
    sql = "SELECT id, created_at, label, confidence, sender, preview, text, keywords FROM scans WHERE 1=1"
    values: list[object] = []
    if label != "all":
        sql += " AND label = ?"
        values.append(label)
    if q.strip():
        sql += " AND (sender LIKE ? OR preview LIKE ?)"
        values.extend([f"%{q.strip()}%", f"%{q.strip()}%"])
    sql += " ORDER BY id DESC LIMIT ?"
    values.append(limit)
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(sql, values).fetchall()
    return [scan_dict(row) for row in rows]


@app.get("/api/model")
def model_metrics():
    return classifier.metrics


@app.get("/api/dashboard")
def dashboard():
    today = date.today().isoformat()
    with sqlite3.connect(DB_PATH) as connection:
        total, spam, ham, phishing, today_total = connection.execute(
            "SELECT COUNT(*), SUM(label='spam'), SUM(label='ham'), SUM(label='phishing'), SUM(substr(created_at,1,10)=?) FROM scans", (today,)
        ).fetchone()
        recent_rows = connection.execute("SELECT id, created_at, label, confidence, sender, preview, text, keywords FROM scans ORDER BY id DESC LIMIT 6").fetchall()
        daily_rows = connection.execute(
            "SELECT substr(created_at,1,10), COUNT(*), SUM(label='spam'), SUM(label='ham'), SUM(label='phishing') FROM scans WHERE created_at >= ? GROUP BY 1 ORDER BY 1",
            ((date.today() - timedelta(days=6)).isoformat(),),
        ).fetchall()
    daily_map = {row[0]: row[1:] for row in daily_rows}
    daily = []
    for offset in range(6, -1, -1):
        day = date.today() - timedelta(days=offset)
        counts = daily_map.get(day.isoformat(), (0, 0, 0, 0))
        daily.append({"date": day.isoformat(), "total": counts[0], "spam": counts[1] or 0, "ham": counts[2] or 0, "phishing": counts[3] or 0})
    return {
        "total_scans": total or 0, "spam_count": spam or 0, "ham_count": ham or 0, "phishing_count": phishing or 0,
        "today_scans": today_total or 0, "spam_share": round((spam or 0) / total, 3) if total else 0,
        "daily": daily, "recent": [scan_dict(row) for row in recent_rows], "model": classifier.metrics,
    }


@app.post("/api/dataset")
async def upload_dataset(file: UploadFile = File(...), model: str = Form(default="naive_bayes")):
    if model not in MODEL_OPTIONS:
        raise HTTPException(status_code=400, detail="Choose naive_bayes, logistic_regression, or linear_svm.")
    if file.size and file.size > 20_000_000:
        raise HTTPException(status_code=413, detail="Dataset must be smaller than 20 MB.")
    payload = await file.read()
    try:
        records = read_records(payload, file.filename or "dataset.csv")
        updated = Classifier(records, file.filename or "Uploaded dataset", model)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    global classifier
    with classifier_lock:
        classifier = updated
    DATA_PATH.write_bytes(payload)
    return {"message": "Model retrained successfully.", "metrics": classifier.metrics}


if FRONTEND_PATH.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")