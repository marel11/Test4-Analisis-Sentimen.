# Praktikum Analisis Sentimen Mitigasi Bencana dengan Neural Network

## Penting: status data

`tweet_data.csv` sengaja baru berisi **header**. Tidak ada tweet/komentar rekaan. Anda perlu mengambil **minimal 300 data asli** melalui API resmi atau CSV yang sah sebelum mendapat hasil model. Skrip crawler mencoba mengumpulkan 350 data supaya ada cadangan untuk duplikasi/pembersihan. **Nama file `tweet_data.csv` hanya mengikuti penamaan tugas**; bila sumbernya YouTube, sebut secara jujur sebagai *komentar YouTube*, **bukan tweet**.

## 1. Struktur proyek

- `Praktikum_Sentimen_Mitigasi_Bencana.ipynb` — notebook Jupyter/Google Colab: preprocessing, labeling, grafik, WordCloud, TF-IDF + MLP, evaluasi, prediksi, ekspor hasil.
- `app_sentimen_nn.py` — dashboard Streamlit (dataset, diagram batang/pie, tiga WordCloud, MLP, metrik, confusion matrix, prediksi baru).
- `analisis_sentimen.py` — preprocessing, label berbasis kata, TF-IDF + MLP, klasifikasi.
- `crawl_x.py` — kolektor posting X/Twitter (akses X API diperlukan).
- `crawl_youtube.py` — kolektor komentar video YouTube (API key diperlukan, alternatif jika sumber diizinkan dosen).
- `.env.example` — contoh konfigurasi kunci API tanpa membocorkan token.
- `tweet_data.csv` — template CSV (belum ada baris data nyata).
- `Laporan_Praktikum_Template.docx` — kerangka laporan yang belum diisi hasil eksperimen.
- `output/` — folder untuk menyimpan screenshot dashboard bila sudah ada.

## 2. Instalasi MacBook / Linux

Buka terminal di folder proyek, lalu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell (aktivasi lingkungan):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

## 3A. Ambil X/Twitter (sesuai instruksi langkah 1)

1. Dapatkan akses API dari pengembang X (`https://developer.x.com/` atau `https://docs.x.com/`). Hak akses dan biaya/kuota tergantung paket API Anda.
2. Isi `X_BEARER_TOKEN=...` pada `.env`.
3. Jalankan:

```bash
python crawl_x.py --target 350
```

Pencarian default: "mitigasi bencana", "kesiapsiagaan bencana", "peringatan dini", atau "evakuasi bencana", bahasa Indonesia, tanpa repost.

Jika data belum mencapai 300, coba kata kunci lain, cek jendela waktu dan kuota API; jangan menambahkan data palsu.

## 3B. Alternatif: komentar YouTube (jika sumber lain diperbolehkan)

1. Buat API key di Google Cloud Console, aktifkan **YouTube Data API v3**.
2. Isi `YOUTUBE_API_KEY=...` di `.env`.
3. Jalankan:

```bash
python crawl_youtube.py --target 350
```

Skrip memilih video dari pencarian mitigasi/kesiapsiagaan/edukasi bencana dan membaca komentar tingkat atas menggunakan API. Beberapa video bisa menutup komentar. Periksa manual relevansi komentar karena tidak semua komentar otomatis membicarakan topik. File tetap dinamai `tweet_data.csv` demi kompatibilitas dashboard; beri keterangan sumber **YouTube** dalam laporan.

API komentar mendukung hingga 100 item/halaman dan `nextPageToken` untuk menelusuri halaman berikutnya (dokumentasi: `https://developers.google.com/youtube/data/reference/rest/v3/commentThreads/list`).

## 3C. CSV sendiri (tanpa API)

Bila Anda sudah punya CSV hasil crawling legal, tempatkan dengan nama `tweet_data.csv`. Header wajib:

```csv
created_at,username,full_text
```

Kolom `source` dan `source_url` sangat dianjurkan untuk audit asal data. Hindari membagikan identitas atau data sensitif yang tidak diperlukan. **Jangan** gunakan CSV contoh sintetis sebagai penelitian nyata.

## 4. Jalankan dashboard

```bash
streamlit run app_sentimen_nn.py
```

Buka `http://localhost:8501`. Dashboard juga bisa menerima CSV melalui menu unggah di sidebar.

**Panel dashboard**: Dataset → Distribusi (bar & pie) → WordCloud positif/netral/negatif → Evaluasi MLP → Prediksi teks manual dan probabilitas.

Rasio split 80:20, stratifikasi kelas, TF-IDF maksimum 5000 fitur (1–2 gram), satu hidden layer 100 neuron, ReLU, Adam, `max_iter=250`, `random_state=42`. Metrik presisi/recall/F1 memakai **weighted average** agar tetap terdefinisi pada kelas yang tak seimbang.

Jika hanya satu kelas, atau ada kelas dengan kurang dari dua baris, model tidak bisa diuji dengan strategi ini: tambah data nyata dari kategori lain dan cek leksikon.


## 4A. Menjalankan kode notebook `.ipynb`

- **VS Code:** buka `Praktikum_Sentimen_Mitigasi_Bencana.ipynb`, pilih interpreter `.venv`, lalu **Run All** setelah `tweet_data.csv` berisi data asli.
- **Google Colab:** unggah notebook dan `tweet_data.csv` hasil crawling, kemudian jalankan sel dari atas. Notebook otomatis meminta unggah CSV bila belum ditemukan. Jika library kurang, aktifkan baris `%pip install` di sel awal.
- Notebook berdiri sendiri; tidak bergantung pada import `analisis_sentimen.py`. Dashboard tetap dijalankan terpisah dengan Streamlit. Hasil notebook disimpan di `output/`.

## 5. Simpan hasil untuk laporan

- Tab Dataset → unduh `tweet_data_berlabel.csv`.
- Tab Evaluasi → unduh `hasil_evaluasi_nn.csv` dan `prediksi_data_uji.csv`.
- Ambil screenshot **dari aplikasi yang benar-benar berjalan**, lalu tempelkan ke laporan untuk: dataset, bar/pie, WordCloud 3 sentimen, metrik, confusion matrix, prediksi baru.
- Gunakan template laporan DOCX; ganti setiap `[ISI ...]` dengan hasil nyata. Catat tanggal akuisisi, platform, rentang tanggal, jumlah data awal dan setelah deduplikasi.

## 6. Batasan interpretasi ilmiah

**Label pelatihan otomatis dibuat dari kata positif/negatif pada teks yang sama.** Maka skor tinggi pada data testing utamanya menunjukkan kemampuan MLP meniru label rule-based, **bukan** validitas terhadap penilaian manusia. Evaluasi ideal: anotasi manual sampel (mis. 50 data), perbandingan dengan label otomatis, dan pembahasan ironi, konteks, negasi, serta bias kategori. Kata seperti “gempa”/“banjir” mengindikasikan peristiwa negatif, tetapi bukan selalu opini negatif. Nilai `predict_proba` bukan jaminan probabilitas terkalibrasi.

## 7. Deploy ke GitHub + Streamlit Cloud

Setelah program berjalan dan dataset publiknya aman dibagikan:

```bash
git init -b main
git add .
git commit -m "Praktikum analisis sentimen mitigasi bencana dengan MLP"
git remote add origin https://github.com/USERNAME/NAMA-REPOSITORY.git
git push -u origin main
```

Ganti URL dengan repo GitHub yang **benar-benar Anda buat**. Jika `origin` sudah ada, pakai `git remote -v` lalu `git remote set-url origin URL_BARU`; jangan langsung force push.

Pada `https://share.streamlit.io/`, pilih repositori, cabang `main`, dan main file `app_sentimen_nn.py`. Bila dashboard mengandalkan CSV lokal, file tersebut harus tersedia di repo atau pengguna mengunggahnya; jangan unggah `.env`. Instalasi menggunakan `requirements.txt`.

## Checklist kesesuaian tugas

1. Ambil >=300 data asli (`tweet_data.csv`)
2. App Streamlit (`app_sentimen_nn.py`)
3. Cleansing → `clean_text`
4. Rule-based labeling → `sentimen`
5. Diagram batang dan pie
6. WordCloud tiga kelas
7. Split 80:20, TF-IDF, MLP 100 neuron ReLU Adam
8. Akurasi, presisi, recall, F1, confusion matrix
9. Prediksi manual dan probabilitas + emoji
10. Isi template laporan dan screenshot dashboard nyata

