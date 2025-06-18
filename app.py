import streamlit as st
import pandas as pd
import joblib
import os

# DEBUG: untuk memastikan versi terbaru terbaca
st.write("🔄 Versi terbaru 2025 ✅")

# Load model dan scaler
kmeans = joblib.load('model_kmeans.pkl')
scaler = joblib.load('scaler.pkl')
label_map = joblib.load('label_map.pkl')

# Konfigurasi halaman
st.set_page_config(page_title="Kualitas Air", layout="centered")

st.title("💧 Pemantauan Kualitas Air Aquaponik")

with st.form("input_form"):
    st.subheader("📝 Input Data")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, format="%.1f")
    suhu = st.number_input("Suhu (°C)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
    tds = st.number_input("TDS (mg/L)", min_value=0.0, step=0.1, format="%.1f")

    submitted = st.form_submit_button("SUBMIT")

if submitted:
    # Prediksi dari model
    input_data = pd.DataFrame([[ph, suhu, tds]], columns=['ph', 'temperature', 'tds'])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    kualitas = label_map[cluster]

    # Warna status
    warna = {
        'Baik': 'green',
        'Cukup': 'orange',
        'Buruk': 'red'
    }

    # Tampilkan status kualitas air
    st.markdown(f"""
    <div style="padding: 1em; background-color: {warna[kualitas]}; color: white; border-radius: 8px; text-align: center;">
        <h4>Status Kualitas Air:</h4>
        <h2><b>{kualitas.upper()}</b></h2>
    </div>
    """, unsafe_allow_html=True)

    # Tampilkan data input
    st.write(f"**pH:** {ph}")
    st.write(f"**Suhu:** {suhu} °C")
    st.write(f"**TDS:** {tds} mg/L")

    # Notifikasi + Rekomendasi
    if kualitas == 'Buruk':
        st.error("⚠️ Kualitas air **BURUK** berdasarkan hasil model.")
        
        rekomendasi = []

        if ph < 6.5:
            rekomendasi.append("- pH terlalu rendah. Tambahkan kapur (CaCO₃) atau air basa.")
        elif ph > 8.5:
            rekomendasi.append("- pH terlalu tinggi. Tambahkan asam humat atau ganti sebagian air.")

        if suhu < 20:
            rekomendasi.append("- Suhu terlalu rendah. Gunakan pemanas atau pindahkan ke lokasi lebih hangat.")
        elif suhu > 30:
            rekomendasi.append("- Suhu terlalu tinggi. Tambahkan peneduh atau sistem pendingin.")

        if tds > 1000:
            rekomendasi.append("- TDS terlalu tinggi. Kurangi pakan ikan dan ganti sebagian air.")

        if rekomendasi:
            st.markdown("#### 💡 Rekomendasi Perbaikan:")
            for item in rekomendasi:
                st.markdown(f"- {item}")

    elif kualitas == 'Cukup':
        st.warning("⚠️ Kualitas air **CUKUP** berdasarkan hasil model.")
    elif kualitas == 'Baik':
        st.success("✅ Kualitas air **BAIK** berdasarkan hasil model.")
