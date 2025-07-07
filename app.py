"""
Human Voice Gender Classifier - Streamlit Application
A robust application for classifying gender from voice data using machine learning models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import io
import traceback
from typing import Optional, Tuple, List, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🎙️ Human Voice Gender Classifier", 
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class VoiceGenderClassifier:
    """Main class for voice gender classification."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.tf_model = None
        self.feature_names = [
            'mean_pitch', 'mean_centroid', 'mean_bandwidth', 
            'zero_crossing', 'rms'
        ] + [f'mfcc_mean_{i}' for i in range(13)] + [f'mfcc_std_{i}' for i in range(13)]
        
    def load_models(self) -> bool:
        """Load pre-trained models with error handling."""
        try:
            # Import required libraries
            import joblib
            from tensorflow.keras.models import load_model
            
            # Load models
            model_paths = {
                'rf_model': '/content/best_model.pkl',
                'xgb_model': '/content/advanced_best_model.pkl',
                'scaler': '/content/scaler.pkl',
                'tf_model': 'tf_gender_model.h5'
            }
            
            for name, path in model_paths.items():
                if not Path(path).exists():
                    st.error(f"Model file not found: {path}")
                    return False
            
            self.models['rf'] = joblib.load(model_paths['rf_model'])
            self.models['xgb'] = joblib.load(model_paths['xgb_model'])
            self.scaler = joblib.load(model_paths['scaler'])
            self.tf_model = load_model(model_paths['tf_model'])
            
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def extract_audio_features(self, audio_data: bytes, sr: int = 22050) -> Optional[np.ndarray]:
        """Extract features from audio data."""
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(io.BytesIO(audio_data), sr=sr)
            
            if len(y) == 0:
                st.error("Audio file is empty or corrupted.")
                return None
            
            # Extract features
            features = []
            
            # Pitch (fundamental frequency)
            try:
                f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
                mean_pitch = np.nanmean(f0[f0 > 0]) if np.any(f0 > 0) else 0
                features.append(mean_pitch)
            except:
                features.append(0)
            
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
            st.error(f"Error extracting features: {str(e)}")
            return None
    
    def create_audio_visualizations(self, audio_data: bytes) -> Dict[str, plt.Figure]:
        """Create audio visualizations."""
        try:
            import librosa
            import librosa.display
            
            y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
            visualizations = {}
            
            # Waveform
            fig1, ax1 = plt.subplots(figsize=(12, 4))
            ax1.plot(np.linspace(0, len(y)/sr, len(y)), y, color='steelblue', linewidth=0.5)
            ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            visualizations['waveform'] = fig1
            
            # Spectrogram
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='viridis')
            ax2.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
            fig2.colorbar(img, ax=ax2, format='%+2.0f dB')
            plt.tight_layout()
            visualizations['spectrogram'] = fig2
            
            # Pitch contour
            try:
                f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
                times = librosa.times_like(f0, sr=sr)
                fig3, ax3 = plt.subplots(figsize=(12, 4))
                ax3.plot(times, f0, color='crimson', linewidth=1.5)
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Pitch (Hz)')
                ax3.set_title('Pitch Contour', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 400)
                plt.tight_layout()
                visualizations['pitch'] = fig3
            except:
                pass
            
            return visualizations
            
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
            return {}
    
    def predict_gender(self, features: np.ndarray, model_type: str) -> Tuple[str, float]:
        """Predict gender from features."""
        try:
            # Scale features
            if self.scaler is None:
                raise ValueError("Scaler not loaded")
            
            # Ensure features match scaler dimensions
            if features.shape[1] != len(self.scaler.mean_):
                # Pad or trim features to match expected dimensions
                expected_dim = len(self.scaler.mean_)
                current_dim = features.shape[1]
                
                if current_dim < expected_dim:
                    # Pad with zeros
                    padding = np.zeros((features.shape[0], expected_dim - current_dim))
                    features = np.concatenate([features, padding], axis=1)
                elif current_dim > expected_dim:
                    # Trim features
                    features = features[:, :expected_dim]
            
            X_scaled = self.scaler.transform(features)
            
            # Get model
            model = self.models.get(model_type)
            if model is None:
                raise ValueError(f"Model {model_type} not found")
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            label = "Female" if prediction == 0 else "Male"
            confidence = max(probabilities)
            
            return label, confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return "Unknown", 0.0
    
    def plot_feature_importance(self, model_type: str) -> Optional[plt.Figure]:
        """Plot feature importance for tree-based models."""
        try:
            model = self.models.get(model_type)
            if model is None or not hasattr(model, 'feature_importances_'):
                return None
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(indices)), importances[indices], color='skyblue', alpha=0.8)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}' for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error plotting feature importance: {str(e)}")
            return None

def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Human Voice Gender Classifier</h1>
        <p>Advanced AI-powered voice analysis for gender classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    classifier = VoiceGenderClassifier()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Select Classification Model",
            ("Random Forest", "XGBoost", "Neural Network"),
            help="Choose the machine learning model for classification"
        )
        
        model_map = {
            "Random Forest": "rf",
            "XGBoost": "xgb",
            "Neural Network": "tf"
        }
        
        selected_model = model_map[model_choice]
        
        st.markdown("---")
        
        # Information
        st.markdown("""
        ### 📊 Supported Features
        - **Pitch Analysis**: Fundamental frequency extraction
        - **Spectral Features**: Centroid, bandwidth, zero-crossing rate
        - **MFCCs**: 13 Mel-frequency cepstral coefficients
        - **Audio Visualizations**: Waveform, spectrogram, pitch contour
        """)
        
        st.markdown("---")
        
        # File format info
        st.markdown("""
        ### 📁 Supported Formats
        - **CSV**: Pre-extracted features
        - **Audio**: WAV, MP3 files
        - **Sample Rate**: Automatically handled
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 CSV Analysis", "🎵 Audio Analysis", "📈 Model Info"])
    
    with tab1:
        st.header("📊 CSV Feature Analysis")
        st.markdown("Upload a CSV file with pre-extracted audio features for batch classification.")
        
        uploaded_csv = st.file_uploader(
            "Choose a CSV file", 
            type=["csv"],
            help="CSV should contain audio features as columns"
        )
        
        if uploaded_csv:
            try:
                # Load CSV
                data = pd.read_csv(uploaded_csv)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Data Preview")
                    st.dataframe(data.head(10), use_container_width=True)
                
                with col2:
                    st.subheader("Dataset Info")
                    st.write(f"**Rows:** {len(data)}")
                    st.write(f"**Columns:** {len(data.columns)}")
                    st.write(f"**Memory Usage:** {data.memory_usage().sum() / 1024:.1f} KB")
                
                # Load models and make predictions
                if st.button("🔍 Analyze CSV", type="primary"):
                    with st.spinner("Loading models and making predictions..."):
                        if classifier.load_models():
                            try:
                                # Prepare features
                                X_scaled = classifier.scaler.transform(data)
                                model = classifier.models[selected_model]
                                
                                # Make predictions
                                predictions = model.predict(X_scaled)
                                probabilities = model.predict_proba(X_scaled)
                                
                                # Create results
                                results = data.copy()
                                results['Predicted_Gender'] = ['Female' if p == 0 else 'Male' for p in predictions]
                                results['Confidence'] = probabilities.max(axis=1)
                                
                                # Display results
                                st.success("✅ Analysis Complete!")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Total Predictions", len(results))
                                    st.metric("Average Confidence", f"{results['Confidence'].mean():.3f}")
                                
                                with col2:
                                    gender_counts = results['Predicted_Gender'].value_counts()
                                    st.metric("Male Predictions", gender_counts.get('Male', 0))
                                    st.metric("Female Predictions", gender_counts.get('Female', 0))
                                
                                # Results table
                                st.subheader("Prediction Results")
                                st.dataframe(results, use_container_width=True)
                                
                                # Download results
                                csv_output = results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv_output,
                                    file_name="gender_predictions.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                st.error("Please check that your CSV contains the expected features.")
                        else:
                            st.error("Could not load models. Please check model files.")
                            
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
    
    with tab2:
        st.header("🎵 Audio Analysis")
        st.markdown("Upload an audio file for real-time feature extraction and gender classification.")
        
        uploaded_audio = st.file_uploader(
            "Choose an audio file", 
            type=["wav", "mp3", "m4a"],
            help="Upload a clear voice recording for best results"
        )
        
        if uploaded_audio:
            # Audio player
            st.audio(uploaded_audio, format="audio/wav")
            
            # Analysis button
            if st.button("🔍 Analyze Audio", type="primary"):
                with st.spinner("Analyzing audio... This may take a moment."):
                    # Load models
                    if classifier.load_models():
                        try:
                            # Get audio data
                            audio_data = uploaded_audio.read()
                            uploaded_audio.seek(0)  # Reset file pointer
                            
                            # Extract features
                            features = classifier.extract_audio_features(audio_data)
                            
                            if features is not None:
                                # Make prediction
                                gender, confidence = classifier.predict_gender(features, selected_model)
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("🎯 Classification Results")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Predicted Gender", gender)
                                
                                with col2:
                                    st.metric("Confidence", f"{confidence:.3f}")
                                
                                with col3:
                                    confidence_pct = confidence * 100
                                    if confidence_pct > 80:
                                        st.success(f"High Confidence ({confidence_pct:.1f}%)")
                                    elif confidence_pct > 60:
                                        st.warning(f"Medium Confidence ({confidence_pct:.1f}%)")
                                    else:
                                        st.error(f"Low Confidence ({confidence_pct:.1f}%)")
                                
                                # Audio visualizations
                                st.markdown("---")
                                st.subheader("📊 Audio Analysis Visualizations")
                                
                                visualizations = classifier.create_audio_visualizations(audio_data)
                                
                                if 'waveform' in visualizations:
                                    st.pyplot(visualizations['waveform'])
                                
                                if 'spectrogram' in visualizations:
                                    st.pyplot(visualizations['spectrogram'])
                                
                                if 'pitch' in visualizations:
                                    st.pyplot(visualizations['pitch'])
                                
                                # Feature importance
                                if selected_model in ['rf', 'xgb']:
                                    importance_fig = classifier.plot_feature_importance(selected_model)
                                    if importance_fig:
                                        st.subheader("🔍 Feature Importance")
                                        st.pyplot(importance_fig)
                                
                            else:
                                st.error("Could not extract features from audio file.")
                                
                        except Exception as e:
                            st.error(f"Error during audio analysis: {str(e)}")
                            st.error("Please ensure the audio file is valid and contains clear speech.")
                    else:
                        st.error("Could not load models. Please check model files.")
    
    with tab3:
        st.header("📈 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Available Models")
            
            model_info = {
                "Random Forest": {
                    "Type": "Ensemble Learning",
                    "Strengths": "Robust, handles overfitting well",
                    "Best for": "General purpose classification"
                },
                "XGBoost": {
                    "Type": "Gradient Boosting",
                    "Strengths": "High performance, feature importance",
                    "Best for": "Competitive accuracy"
                },
                "Neural Network": {
                    "Type": "Deep Learning",
                    "Strengths": "Complex pattern recognition",
                    "Best for": "Large datasets"
                }
            }
            
            for model_name, info in model_info.items():
                with st.expander(f"{model_name} Details"):
                    for key, value in info.items():
                        st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("📊 Feature Extraction")
            
            feature_categories = {
                "Pitch Features": ["Fundamental frequency (F0)", "Pitch variations"],
                "Spectral Features": ["Spectral centroid", "Spectral bandwidth", "Zero crossing rate"],
                "Temporal Features": ["RMS energy", "Signal duration"],
                "Cepstral Features": ["13 MFCCs", "MFCC statistics (mean, std)"]
            }
            
            for category, features in feature_categories.items():
                with st.expander(f"{category}"):
                    for feature in features:
                        st.write(f"• {feature}")
        
        st.markdown("---")
        st.subheader("🔧 Technical Notes")
        
        st.markdown("""
        **Audio Processing:**
        - Sample rate: Automatically detected and standardized
        - Feature extraction: Librosa library
        - Preprocessing: Z-score normalization
        
        **Model Training:**
        - Features: 31-dimensional feature vectors
        - Cross-validation: 5-fold stratified
        - Evaluation metrics: Accuracy, precision, recall, F1-score
        
        **Performance Tips:**
        - Use clear, noise-free audio recordings
        - Ensure adequate recording length (3+ seconds)
        - Avoid background noise and multiple speakers
        """)

if __name__ == "__main__":
    main()
