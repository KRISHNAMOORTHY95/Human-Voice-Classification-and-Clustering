# app.py
!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
tf_model = load_model("tf_gender_model.h5")


# Load models
rf_model = joblib.load("/content/best_model.pkl")
xgb_model = joblib.load("/content/advanced_best_model.pkl")
scaler = joblib.load("/content/scaler.pkl")

st.set_page_config(page_title="🎙️ Human Voice Gender Classifier", layout="wide")
st.title("🎙️ Human Voice Gender Classifier")

st.markdown("""
Upload either:
- A CSV file with extracted features
- OR
- A WAV/MP3 file for live feature extraction
""")

# Model selector
model_choice = st.selectbox(
    "Select Classification Model",
    ("Random Forest", "XGBoost")
)

if model_choice == "Random Forest":
    model = rf_model
else:
    model = xgb_model

# CSV uploader
uploaded_csv = st.file_uploader("Upload CSV with features", type=["csv"])

# Audio uploader
uploaded_audio = st.file_uploader("Or upload a WAV/MP3 file", type=["wav", "mp3"])

if uploaded_csv:
    data = pd.read_csv(uploaded_csv)
    st.write("CSV preview:")
    st.dataframe(data.head())

    X_scaled = scaler.transform(data)
    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    readable = ["Female" if p==0 else "Male" for p in prediction]
    result_df = data.copy()
    result_df["Predicted Gender"] = readable
    result_df["Confidence"] = probabilities.max(axis=1)

    st.success("✅ Predictions complete for CSV.")
    st.dataframe(result_df)

    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Results", csv_out, "predictions.csv", "text/csv")

elif uploaded_audio:
    st.audio(uploaded_audio)

    # librosa
    y, sr = librosa.load(uploaded_audio, sr=None)

    # Waveform
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(y, color="steelblue")
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Spectrogram
    st.subheader("Spectrogram")
    import librosa.display
    fig2, ax2 = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    st.pyplot(fig2)

    # Pitch contour
    st.subheader("Pitch Contour")
    try:
        f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)
        times = librosa.times_like(f0, sr=sr)
        fig3, ax3 = plt.subplots()
        ax3.plot(times, f0, color="crimson")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pitch (Hz)")
        ax3.set_title("Pitch Contour")
        st.pyplot(fig3)
    except:
        st.warning("Could not extract pitch contour.")

    # feature extraction
    try:
        mean_pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr).mean()
    except:
        mean_pitch = 0
    mean_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    mean_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    zero_crossing = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y).mean()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)
    mfcc_stds = mfccs.std(axis=1)

    feature_vector = [
        mean_pitch,
        mean_centroid,
        mean_bandwidth,
        zero_crossing,
        rms
    ]
    feature_vector.extend(mfcc_means)
    feature_vector.extend(mfcc_stds)

    while len(feature_vector) < scaler.mean_.shape[0]:
        feature_vector.append(0)

    X_input = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    pred = model.predict(X_scaled)[0]
    label = "Female" if pred==0 else "Male"

    st.success(f"✅ Predicted gender from audio: **{label}**")

    # feature importance
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        feature_names = data.columns if uploaded_csv else [f"f{i}" for i in range(X_scaled.shape[1])]
        imp = model.feature_importances_
        importances = pd.Series(imp, index=feature_names).sort_values(ascending=False).head(10)
        fig4, ax4 = plt.subplots()
        ax4.barh(importances.index, importances.values, color="skyblue")
        ax4.set_title("Top 10 Feature Importances")
        st.pyplot(fig4)

else:
    st.info("Upload a CSV or audio file to start.")
