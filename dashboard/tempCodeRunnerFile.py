import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================
# Load Data & Models
# =====================
DATA_PATH = "data"

df = pd.read_csv(f"{DATA_PATH}/courses_clean.csv")

tfidf = joblib.load(f"{DATA_PATH}/tfidf_vectorizer.pkl")
cosine_sim = joblib.load(f"{DATA_PATH}/cosine_similarity.pkl")

forecast_df = pd.read_csv(f"{DATA_PATH}/sales_forecast_result.csv")
# =====================
# Helper Function
# =====================
def recommend_course(kategori_input, level_input, top_n=5):
    filtered_df = df[
        df["kategori"].str.lower().str.contains(kategori_input.lower(), na=False) &
        df["level"].str.lower().str.contains(level_input.lower(), na=False)
    ]

    if filtered_df.empty:
        return pd.DataFrame()

    idx = filtered_df.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]
    indices = [i[0] for i in sim_scores]

    return df.loc[indices]

# =====================
# Streamlit Config
# =====================
st.set_page_config(
    page_title="Course Recommendation & Sales Forecasting",
    layout="wide"
)

st.sidebar.title("📊 Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["User View", "Admin View"])

# =====================
# USER VIEW
# =====================
if menu == "User View":
    st.title("🎓 Sistem Rekomendasi Course")

    kategori = st.text_input(
        "Minat / Kategori (contoh: Business, Finance, Data)"
    )

    level = st.selectbox(
        "Level Pembelajaran",
        sorted(df["level"].unique())
    )

    top_n = st.slider("Jumlah Rekomendasi", 1, 10, 5)

    if st.button("🔍 Tampilkan Rekomendasi"):
        hasil = recommend_course(kategori, level, top_n)

        if hasil.empty:
            st.warning("Tidak ada course yang sesuai dengan kriteria.")
        else:
            for _, row in hasil.iterrows():
                with st.expander(row["nama_course"]):
                    st.write(f"📂 **Kategori**: {row['kategori']}")
                    st.write(f"🎯 **Level**: {row['level']}")
                    st.write(f"⭐ **Rating**: {row['rating']}")
                    st.write(f"💰 **Harga**: Rp {row['price']:,}")
                    st.markdown("**🎁 Benefit:**")
                    st.write(row["benefit"])

# =====================
# ADMIN VIEW
# =====================
if menu == "Admin View":
    st.title("📈 Dashboard Sales Forecasting")

    # KPI
    col1, col2, col3 = st.columns(3)

    total_siswa = int(df["jumlah_peserta"].sum())
    avg_rating = round(df["rating"].mean(), 2)
    total_revenue = int((df["jumlah_peserta"] * df["price"]).sum())

    col1.metric("Total Siswa", total_siswa)
    col2.metric("Rata-rata Rating", avg_rating)
    col3.metric("Total Revenue", f"Rp {total_revenue:,}")

    # Forecast Chart
    st.subheader("📉 Grafik Prediksi Penjualan")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        forecast_df["date"],
        forecast_df["predicted_sales"],
        label="Prediksi",
        color="orange"
    )
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Jumlah Siswa")
    ax.set_title("Sales Forecasting")
    ax.legend()

    st.pyplot(fig)

    # Business Insight
    st.subheader("💡 Rekomendasi Bisnis")

    st.markdown("""
    **Strategi Produk**
    - Fokus pada kategori dengan jumlah peserta tinggi
    - Tambah course level Beginner & Intermediate

    **Strategi Pricing**
    - Bundling course per kategori
    - Diskon course mahal dengan rating tinggi

    **Strategi Marketing**
    - Promosi course populer
    - Personalisasi rekomendasi user

    **Strategi Resource**
    - Alokasi instruktur pada course dengan pertumbuhan tinggi
    """)
