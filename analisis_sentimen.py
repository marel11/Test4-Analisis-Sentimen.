"""Fungsi pembersihan teks, label leksikon, pelatihan, dan evaluasi MLP."""
from __future__ import annotations

import html
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

KELAS = ['negatif', 'netral', 'positif']
KATA_POSITIF = {
    'aman', 'selamat', 'siaga', 'siap', 'sigap', 'tanggap', 'cepat', 'baik', 'bagus',
    'efektif', 'berhasil', 'terbantu', 'membantu', 'bantuan', 'peduli', 'terlindungi',
    'perlindungan', 'edukasi', 'antisipasi', 'pulih', 'pemulihan', 'lancar', 'mantap',
    'apresiasi', 'hebat', 'terlatih', 'tertib', 'waspada', 'tenang', 'solid', 'sukses'
}
KATA_NEGATIF = {
    'banjir', 'gempa', 'longsor', 'kebakaran', 'bahaya', 'berbahaya', 'parah',
    'rusak', 'kerusakan', 'terlambat', 'lambat', 'gagal', 'panik', 'takut', 'khawatir',
    'korban', 'meninggal', 'terjebak', 'kacau', 'macet', 'buruk', 'kurang', 'lemah',
    'minim', 'krisis', 'hancur', 'celaka', 'terancam', 'mengungsi', 'bencana', 'darurat'
}
STOPWORDS_ID = {
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'atau', 'pada', 'ini', 'itu',
    'ada', 'adalah', 'akan', 'juga', 'sudah', 'agar', 'karena', 'oleh', 'jadi',
    'sebagai', 'dalam', 'lebih', 'saat', 'para', 'mereka', 'kami', 'kita', 'saya',
    'aku', 'anda', 'tidak', 'bukan', 'masih', 'bisa', 'harus', 'telah', 'tahun',
    'hari', 'jadi', 'sekarang', 'sangat', 'terus', 'sebuah', 'bagi', 'tentang',
    'nya', 'yg', 'ya', 'nih', 'sih', 'dong', 'aja', 'itu', 'tp', 'tapi', 'pun', 'lah'
}


def clean_text(text: str) -> str:
    """Cleansing berurutan: URL, mention, hashtag, angka, simbol, lowercase, spasi."""
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text, flags=re.I)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#[\w]+', ' ', text)
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def label_rule_based(cleaned: str) -> str:
    tokens = cleaned.split()
    pos = sum(token in KATA_POSITIF for token in tokens)
    neg = sum(token in KATA_NEGATIF for token in tokens)
    if pos > neg:
        return 'positif'
    if neg > pos:
        return 'negatif'
    return 'netral'


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = {'created_at', 'username', 'full_text'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError('Kolom CSV belum lengkap: ' + ', '.join(sorted(missing)))
    data = df.copy()
    for col in ['created_at', 'username', 'full_text']:
        data[col] = data[col].fillna('').astype(str)
    data = data[data['full_text'].str.strip().ne('')].copy()
    data['clean_text'] = data['full_text'].map(clean_text)
    data = data[data['clean_text'].str.len() >= 8].copy()
    data = data.drop_duplicates(subset=['clean_text']).reset_index(drop=True)
    data['sentimen'] = data['clean_text'].map(label_rule_based)
    return data


def most_common_tokens(texts: list[str], limit: int = 15) -> pd.DataFrame:
    counter = Counter(
        word for text in texts for word in text.split()
        if len(word) >= 3 and word not in STOPWORDS_ID
    )
    return pd.DataFrame(counter.most_common(limit), columns=['kata', 'jumlah'])


def train_evaluate(data: pd.DataFrame) -> dict:
    if len(data) < 30:
        raise ValueError('Data yang telah dibersihkan kurang dari 30 baris. Tambahkan data asli.')
    counts = data['sentimen'].value_counts()
    if len(counts) < 2:
        raise ValueError('Hanya satu kelas sentimen ditemukan. Perlu minimal dua kelas untuk evaluasi.')
    if counts.min() < 2:
        raise ValueError('Setiap kelas yang ditemukan harus memiliki minimal 2 data agar pembagian stratified berhasil.')
    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['sentimen'], test_size=0.2,
        random_state=42, stratify=data['sentimen']
    )
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(100,), activation='relu', solver='adam',
            max_iter=250, random_state=42, early_stopping=False
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    present_labels = [c for c in KELAS if c in counts.index]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    metrics = {
        'Akurasi': float(accuracy_score(y_test, y_pred)),
        'Presisi (weighted)': float(precision),
        'Recall (weighted)': float(recall),
        'F1-score (weighted)': float(f1),
    }
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    evaluation = pd.DataFrame({
        'teks': X_test.to_numpy(),
        'aktual': y_test.to_numpy(),
        'prediksi': y_pred,
    }).reset_index(drop=True)
    report = classification_report(
        y_test, y_pred, labels=present_labels,
        output_dict=True, zero_division=0
    )
    return dict(model=model, metrics=metrics, cm=cm, labels=present_labels,
                evaluation=evaluation, report=pd.DataFrame(report).T,
                train_n=len(X_train), test_n=len(X_test))


def predict_one(model: Pipeline, user_text: str) -> tuple[str, dict[str, float]]:
    clean = clean_text(user_text)
    if not clean:
        raise ValueError('Masukkan teks yang mengandung kata, bukan hanya tautan atau simbol.')
    pred = str(model.predict([clean])[0])
    proba = model.predict_proba([clean])[0]
    labels = model.named_steps['mlp'].classes_
    return pred, {str(k): float(v) for k, v in zip(labels, proba)}
