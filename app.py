"""
Voice Gender Classifier - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import io
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🎙️ Voice Gender Classifier",
    page_icon="🎙️",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class VoiceClassifier:
    """Simplified voice gender classifier."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_models(self):
        """Load pre-trained model and scaler."""
        try:
            # Try different possible paths
            model_paths = ['best_model.pkl', './models/best_model.pkl']
            scaler_paths = ['scaler.pkl', './models/scaler.pkl']
            
            # Load model
            for path in model_paths:
                if Path(path).exists():
                    self.model = joblib.load(path)
                    break
            
            # Load scaler
            for path in scaler_paths:
                if Path(path).exists():
                    self.scaler = joblib.load(path)
                    break
                    
            return self.model is not None and self.scaler is not None
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def extract_features(self, audio_data):
        """Extract features from audio data."""
        try:
            # Load audio
            y, sr = librosa.load(io.BytesIO(audio_data), sr=22050)
            
            # Extract basic features
            features = []
            
            # Pitch
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            features.append(np.nanmean(f0[f0 > 0]) if np.any(f0 > 0) else 0)
            
            # Spectral features
            features.append(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
            features.append(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
            features.append(librosa.feature.zero_crossing_rate(y).mean())
            features.append(librosa.feature.rms(y=y).mean())
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(mfccs.mean(axis=1))
            features.extend(mfccs.std(axis=1))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None
    
    def predict(self, features):
        """Predict gender from features."""
        try:
            # Scale features
            X_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            gender = "Female" if prediction == 0 else "Male"
            confidence = max(probabilities)
            
            return gender, confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return "Unknown", 0.0

def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Voice Gender Classifier</h1>
        <p>AI-powered voice analysis for gender classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    classifier = VoiceClassifier()
    
    # Load models once
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = classifier.load_models()
        if not st.session_state.models_loaded:
            st.error("⚠️ Could not load models. Please check if model files exist.")
            st.stop()
    
    # Tabs
    tab1, tab2 = st.tabs(["🎵 Audio Analysis", "📊 CSV Analysis"])
    
    with tab1:
        st.header("🎵 Audio Analysis")
        st.write("Upload an audio file for gender classification.")
        
        uploaded_audio = st.file_uploader(
            "Choose an audio file", 
            type=["wav", "mp3", "m4a"]
        )
        
        if uploaded_audio:
            # Show audio player
            st.audio(uploaded_audio)
            
            if st.button("🔍 Analyze Audio", type="primary"):
                with st.spinner("Analyzing..."):
                    # Get audio data
                    audio_data = uploaded_audio.read()
                    
                    # Extract features
                    features = classifier.extract_features(audio_data)
                    
                    if features is not None:
                        # Make prediction
                        gender, confidence = classifier.predict(features)
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Gender", gender)
                        with col2:
                            st.metric("Confidence", f"{confidence:.3f}")
                        
                        # Confidence indicator
                        confidence_pct = confidence * 100
                        if confidence_pct > 80:
                            st.success(f"High Confidence ({confidence_pct:.1f}%)")
                        elif confidence_pct > 60:
                            st.warning(f"Medium Confidence ({confidence_pct:.1f}%)")
                        else:
                            st.error(f"Low Confidence ({confidence_pct:.1f}%)")
    
    with tab2:
        st.header("📊 CSV Analysis")
        st.write("Upload a CSV file with audio features for batch classification.")
        
        uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_csv:
            try:
                # Load CSV
                data = pd.read_csv(uploaded_csv)
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                st.write(f"Dataset shape: {data.shape}")
                
                if st.button("🔍 Analyze CSV", type="primary"):
                    with st.spinner("Analyzing..."):
                        # Scale features
                        X_scaled = classifier.scaler.transform(data)
                        
                        # Make predictions
                        predictions = classifier.model.predict(X_scaled)
                        probabilities = classifier.model.predict_proba(X_scaled)
                        
                        # Create results
                        results = data.copy()
                        results['Predicted_Gender'] = ['Female' if p == 0 else 'Male' for p in predictions]
                        results['Confidence'] = probabilities.max(axis=1)
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Predictions", len(results))
                        with col2:
                            st.metric("Average Confidence", f"{results['Confidence'].mean():.3f}")
                        
                        # Results table
                        st.subheader("Results")
                        st.dataframe(results)
                        
                        # Download button
                        csv_output = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_output,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** For best results, use clear audio recordings without background noise.")

if __name__ == "__main__":
    main()
