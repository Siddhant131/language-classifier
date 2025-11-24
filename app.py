import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from deep_translator import GoogleTranslator


MODEL_PATH = "model_checkpoint.joblib"
VECTORIZER_WORD_PATH = "vectorizer_word.joblib"
VECTORIZER_CHAR_PATH = "vectorizer_char.joblib"
ENCODER_PATH = "label_encoder.joblib"
TRAIN_DATA_PATH = "train_updated.csv"  
TEST_DATA_PATH = "test_updated.csv"   

st.set_page_config(page_title="Language ID & Translation", layout="centered")


def load_and_preprocess_data(filepath):
    """Loads and preprocesses the data."""
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return None
    
    data = pd.read_csv(filepath)
    data['cleaned_text'] = data['text'].apply(lambda x: x.lower())
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return data

def train_model():
    with st.spinner('Training model... This may take a moment.'):
       
        data_identify = load_and_preprocess_data(TRAIN_DATA_PATH)
        if data_identify is None:
            return None, None, None, None

        x_identify = data_identify["cleaned_text"]
        y_identify = data_identify["labels"]

        label_encoder = LabelEncoder()
        y_identify_encoded = label_encoder.fit_transform(y_identify)

        tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000, min_df=5)
        tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=10000, min_df=5)

        X_word = tfidf_word.fit_transform(x_identify)
        X_char = tfidf_char.fit_transform(x_identify)

        X_combined = np.hstack((X_word.toarray(), X_char.toarray()))

        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_identify_encoded, test_size=0.2, random_state=42, stratify=y_identify_encoded
        )

        svm_model = LinearSVC(C=1.0, random_state=42)
        svm_model.fit(X_train, y_train)
        svm_preds = svm_model.predict(X_val)
        val_acc = accuracy_score(y_val, svm_preds)
        st.success(f"Validation accuracy (SVM): {val_acc:.4f}")

        final_model = LinearSVC(C=1.0, random_state=42)
        final_model.fit(X_combined, y_identify_encoded)

        joblib.dump(final_model, MODEL_PATH)
        joblib.dump(tfidf_word, VECTORIZER_WORD_PATH)
        joblib.dump(tfidf_char, VECTORIZER_CHAR_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)
        
        st.success("Model trained and saved successfully!")
        return final_model, tfidf_word, tfidf_char, label_encoder

@st.cache_resource
def load_resources():
    
    if (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_WORD_PATH) and 
        os.path.exists(VECTORIZER_CHAR_PATH) and os.path.exists(ENCODER_PATH)):
        
        model = joblib.load(MODEL_PATH)
        vec_word = joblib.load(VECTORIZER_WORD_PATH)
        vec_char = joblib.load(VECTORIZER_CHAR_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, vec_word, vec_char, encoder
    else:
        st.warning("Saved model not found. Starting training process...")
        return train_model()

def identify_language(text, model, vec_word, vec_char, encoder):
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    text_word_features = vec_word.transform([cleaned_text])
    text_char_features = vec_char.transform([cleaned_text])
    
    text_combined = np.hstack((text_word_features.toarray(), text_char_features.toarray()))
    
    predicted_label = model.predict(text_combined)
    predicted_language = encoder.inverse_transform(predicted_label)
    
    return predicted_language[0]

# --- Streamlit UI ---

st.title("🌍 Language Identifier & Translator")
st.write("Enter text below to identify its language and translate it.")


model, vec_word, vec_char, encoder = load_resources()

if model is not None:
    
    user_input = st.text_area("Enter text:", height=150)

    if st.button("Identify Language"):
        if user_input.strip():
            lang = identify_language(user_input, model, vec_word, vec_char, encoder)
            st.session_state['detected_lang'] = lang
            st.session_state['original_text'] = user_input
            st.success(f"**Identified Language:** {lang}")
        else:
            st.warning("Please enter some text.")

    # Translation Section
    if 'detected_lang' in st.session_state and 'original_text' in st.session_state:
        st.divider()
        st.subheader("Translation")
        
        lang_map = {
            'English': 'en', 'French': 'fr', 'Spanish': 'es', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Hindi': 'hi', 'Chinese': 'zh-CN',
            'Japanese': 'ja', 'Russian': 'ru', 'Arabic': 'ar', 'Dutch': 'nl',
            'Korean': 'ko', 'Turkish': 'tr', 'Urdu': 'ur', 'Vietnamese': 'vi',
            'Thai': 'th', 'Greek': 'el', 'Bulgarian': 'bg', 'Swahili': 'sw',
            'Polish': 'pl'
        }

        target_options = sorted(list(lang_map.keys()))
        
        col1, col2 = st.columns([3, 1])
        with col1:
            target_lang_name = st.selectbox("Translate to:", target_options, index=0)
        
        with col2:
            st.write("") 
            st.write("") 
            translate_btn = st.button("Translate Now")

        if translate_btn:
            try:
                target_code = lang_map.get(target_lang_name, 'en')
                translator = GoogleTranslator(source='auto', target=target_code)
                translated_text = translator.translate(st.session_state['original_text'])
                
                st.markdown("### Result:")
                st.info(translated_text)
            except Exception as e:
                st.error(f"Translation failed: {e}")

# --- Sidebar: Retrain Option ---
with st.sidebar:
    st.header("Settings")
    if st.button("Force Retrain Model"):
        for f in [MODEL_PATH, VECTORIZER_WORD_PATH, VECTORIZER_CHAR_PATH, ENCODER_PATH]:
            if os.path.exists(f):
                os.remove(f)
        st.cache_resource.clear()
        st.rerun()