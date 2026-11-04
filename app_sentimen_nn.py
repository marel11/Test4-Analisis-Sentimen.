"""Dashboard analisis sentimen mitigasi bencana (Streamlit)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from analisis_sentimen import (
    KELAS, STOPWORDS_ID, clean_text, most_common_tokens,
    prepare_dataset, predict_one, train_evaluate,
)

st.set_page_config(
    page_title='Analisis Sentimen Mitigasi Bencana | Neural Network',
    page_icon='🌏', layout='wide', initial_sidebar_state='expanded'
)
PALET = {'positif': '#219C78', 'netral': '#D0A03A', 'negatif': '#D45E68'}
EMOJI = {'positif': '😊', 'netral': '😐', 'negatif': '😟'}

st.title('🌏 Analisis Sentimen Mitigasi Bencana')
st.caption('Rule-Based Labeling • TF-IDF • Multilayer Perceptron • Streamlit')
st.info('Dataset harus berasal dari unggahan/komentar nyata. Evaluasi mengukur prediksi model terhadap **label otomatis rule-based**, bukan label manusia.')

@st.cache_data(show_spinner=False)
def load_and_prepare(csv_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    return prepare_dataset(pd.read_csv(BytesIO(csv_bytes), encoding='utf-8-sig'))

@st.cache_resource(show_spinner='Melatih Neural Network dan menghitung evaluasi...')
def cached_training(data: pd.DataFrame) -> dict:
    return train_evaluate(data)

st.sidebar.header('Pengaturan data')
uploaded = st.sidebar.file_uploader('Unggah CSV (opsional)', type=['csv'])
if uploaded is not None:
    content = uploaded.getvalue()
    st.sidebar.success('Menggunakan berkas unggahan.')
elif Path('tweet_data.csv').exists():
    content = Path('tweet_data.csv').read_bytes()
    st.sidebar.success('Menggunakan tweet_data.csv lokal.')
else:
    st.warning('Belum ada dataset. Jalankan crawler X/YouTube atau unggah CSV dengan kolom: created_at, username, full_text.')
    st.code('python crawl_x.py --target 350\n# atau\npython crawl_youtube.py --target 350', language='bash')
    st.stop()

try:
    df = load_and_prepare(content)
except Exception as error:
    st.error(f'Dataset tidak dapat diproses: {error}')
    st.stop()

if df.empty:
    st.warning('Tidak ada teks layak analisis setelah pembersihan.')
    st.stop()

counts = df['sentimen'].value_counts().reindex(KELAS, fill_value=0)
if len(df) < 300:
    st.warning(f'Dataset bersih berisi {len(df)} data unik. Ketentuan tugas minimal 300; kumpulkan lebih banyak agar aman setelah duplikasi dihapus.')

for col, val, caption in zip(st.columns(4),
                             [len(df), int(counts['positif']), int(counts['netral']), int(counts['negatif'])],
                             ['Data valid unik', 'Positif', 'Netral', 'Negatif']):
    col.metric(caption, f'{val:,}')

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '📁 Dataset', '📊 Distribusi', '☁️ WordCloud', '🧠 Evaluasi NN', '🔮 Prediksi Baru'
])

with tab1:
    st.subheader('Hasil cleansing dan pelabelan')
    st.dataframe(df, use_container_width=True, height=420)
    st.download_button('⬇️ Unduh data berlabel (CSV)',
                       df.to_csv(index=False).encode('utf-8-sig'),
                       file_name='tweet_data_berlabel.csv', mime='text/csv')
    st.caption('Username dan waktu berasal dari dataset asli. Kolom clean_text dan sentimen dibuat secara otomatis.')

with tab2:
    st.subheader('Distribusi sentimen')
    chart = counts.rename_axis('sentimen').reset_index(name='jumlah')
    left, right = st.columns(2)
    with left:
        fig = px.bar(chart, x='sentimen', y='jumlah', color='sentimen',
                     color_discrete_map=PALET, text='jumlah', title='Jumlah data per kategori')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(chart, names='sentimen', values='jumlah',
                     color='sentimen', color_discrete_map=PALET,
                     hole=0.36, title='Proporsi kategori (%)')
        st.plotly_chart(fig, use_container_width=True)
    max_cat = str(counts.idxmax())
    st.write(f'Kategori terbanyak: **{max_cat}**, yaitu **{counts[max_cat]} dari {len(df)} data ({counts[max_cat] / len(df):.1%})**.')

with tab3:
    st.subheader('WordCloud berdasarkan kategori')
    for sentiment in ['positif', 'netral', 'negatif']:
        st.markdown(f'#### {EMOJI[sentiment]} {sentiment.title()}')
        subset = df.loc[df['sentimen'].eq(sentiment), 'clean_text'].tolist()
        cols = st.columns([3, 2])
        if not subset:
            st.caption('Belum ada teks untuk kategori ini.')
            continue
        text = ' '.join(subset)
        with cols[0]:
            try:
                wc = WordCloud(
                    width=1100, height=420, background_color='white',
                    stopwords=STOPWORDS_ID, colormap={'positif': 'Greens', 'netral': 'Wistia', 'negatif': 'Reds'}[sentiment],
                    collocations=False, max_words=100, min_font_size=10
                ).generate(text)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except ValueError:
                st.caption('Belum cukup kata setelah stopwords dihapus.')
        with cols[1]:
            top = most_common_tokens(subset, 10)
            if not top.empty:
                fig = px.bar(top, x='jumlah', y='kata', orientation='h',
                             title='10 kata teratas', color_discrete_sequence=[PALET[sentiment]])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader('Evaluasi Neural Network')
    st.write('Pembagian data 80:20 dengan stratifikasi kelas, TF-IDF (unigram dan bigram), MLP 100 neuron, ReLU, Adam.')
    try:
        result = cached_training(df[['clean_text', 'sentimen']])
        mcols = st.columns(4)
        for col, (name, value) in zip(mcols, result['metrics'].items()):
            col.metric(name, f'{value:.2%}')
        st.caption(f"Data training: {result['train_n']} | Testing: {result['test_n']} | Random state: 42")
        st.subheader('Confusion Matrix (data testing)')
        cm = result['cm']
        classes = result['labels']
        heat = go.Figure(data=go.Heatmap(
            z=cm, x=classes, y=classes, colorscale='Blues',
            text=cm, texttemplate='%{text}', hovertemplate='Aktual: %{y}<br>Prediksi: %{x}<br>Jumlah: %{z}<extra></extra>'
        ))
        heat.update_layout(xaxis_title='Prediksi', yaxis_title='Aktual', height=420)
        st.plotly_chart(heat, use_container_width=True)
        st.subheader('Classification Report')
        st.dataframe(result['report'].round(3), use_container_width=True)
        st.subheader('Tabel pengujian per teks')
        st.dataframe(result['evaluation'], use_container_width=True)
        metric_csv = pd.DataFrame([result['metrics']]).to_csv(index=False).encode('utf-8-sig')
        st.download_button('⬇️ Unduh metrik evaluasi (CSV)', metric_csv,
                           file_name='hasil_evaluasi_nn.csv', mime='text/csv')
        st.download_button('⬇️ Unduh prediksi data uji (CSV)',
                           result['evaluation'].to_csv(index=False).encode('utf-8-sig'),
                           file_name='prediksi_data_uji.csv', mime='text/csv')
    except ValueError as error:
        st.error(f'Pelatihan belum dapat dilakukan: {error}')
        result = None

with tab5:
    st.subheader('Uji sentimen teks baru')
    if result is None:
        st.warning('Perbaiki data terlebih dahulu agar tersedia model untuk prediksi.')
    else:
        input_text = st.text_area(
            'Tulis unggahan atau komentar mengenai mitigasi bencana',
            placeholder='Contoh: Sosialisasi evakuasi dan sistem peringatan dini sangat membantu warga.'
        )
        if st.button('Prediksi sentimen', type='primary'):
            if not input_text.strip():
                st.warning('Isi teks terlebih dahulu.')
            else:
                try:
                    predicted, probs = predict_one(result['model'], input_text)
                    st.success(f'Sentimen: {EMOJI[predicted]} **{predicted.upper()}**')
                    st.write('Hasil pembersihan:', clean_text(input_text))
                    probability = pd.DataFrame({'sentimen': list(probs), 'probabilitas': list(probs.values())})
                    fig = px.bar(probability, x='sentimen', y='probabilitas',
                                 color='sentimen', color_discrete_map=PALET,
                                 text=probability['probabilitas'].map(lambda x: f'{x:.1%}'),
                                 range_y=[0, 1], title='Probabilitas prediksi kelas')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption('Probabilitas berasal dari model MLP dan tidak menjamin kalibrasi atau kepastian prediksi.')
                except ValueError as error:
                    st.error(str(error))

st.divider()
st.caption('Penelitian praktikum • Interpretasi sentimen memerlukan verifikasi sampel secara manual agar bias leksikon dapat diketahui.')
