"""
SPOTIFY SENTIMENT ANALYSIS - STREAMLIT APP
Binary Classification: Naive Bayes vs SVM
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import io

# Page config
st.set_page_config(
    page_title="Spotify Sentiment Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #191414;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

download_nltk_data()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# ============================================================================
# FUNCTIONS
# ============================================================================

def is_english(text):
    if pd.isna(text) or len(str(text).strip()) < 5:
        return False
    text_lower = str(text).lower()
    markers = ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'this', 'that', 'with', 'for']
    return sum(1 for word in markers if f' {word} ' in f' {text_lower} ') >= 2

def label_binary_with_confidence(text, vader):
    if pd.isna(text):
        return (None, 0.0)
    
    text_lower = str(text).lower()
    
    strong_positive = ['love', 'amazing', 'excellent', 'perfect', 'best', 'fantastic',
                       'incredible', 'outstanding', 'awesome', 'wonderful']
    strong_negative = ['hate', 'terrible', 'awful', 'worst', 'crash', 'broken',
                       'useless', 'garbage', 'disappointed', 'frustrating']
    
    pos_count = sum(1 for word in strong_positive if word in text_lower)
    neg_count = sum(1 for word in strong_negative if word in text_lower)
    
    negations = ['not good', 'not great', 'not working', "doesn't work",
                 "don't like", "can't use", "won't work"]
    has_negation = any(neg in text_lower for neg in negations)
    if has_negation:
        neg_count += 2
    
    vader_score = vader.polarity_scores(text)['compound']
    
    if pos_count >= 2 and vader_score > 0.3:
        return (1, 0.8)
    elif neg_count >= 2 and vader_score < -0.3:
        return (0, 0.8)
    elif pos_count >= 1 and vader_score > 0.5:
        return (1, 0.6)
    elif neg_count >= 1 and vader_score < -0.5:
        return (0, 0.6)
    elif vader_score > 0.3:
        return (1, 0.4)
    elif vader_score < -0.3:
        return (0, 0.4)
    else:
        return (None, 0.0)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'app', 'spotify', 'song', 'music', 'playlist', 'play', 'listen', 'use'}
    stop_words = stop_words.union(custom_stopwords)
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# 🎵 Navigation")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home", "📊 Data Understanding", "🔬 Modeling", "📈 Results", "💡 Insights", "🧪 Test Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Informasi Penelitian")
st.sidebar.info("""
**Judul:** Analisis Sentimen Ulasan Spotify

**Metode:**
- Naive Bayes
- Support Vector Machine (SVM)

**Dataset:** Spotify App Reviews 2022 (Kaggle)

**Peneliti:** Cici Rahma Angriani (1323077)
""")

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">🎵 ANALISIS SENTIMEN APLIKASI SPOTIFY</p>', unsafe_allow_html=True)
    st.markdown("### Menggunakan Metode Naive Bayes dan Support Vector Machine (SVM)")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Latar Belakang")
        st.write("""
        Spotify sebagai platform streaming musik terbesar menghadapi ribuan ulasan pengguna setiap hari. 
        Penelitian ini bertujuan untuk:
        
        - 📌 Menganalisis sentimen ulasan pengguna secara otomatis
        - 📌 Mengidentifikasi masalah dominan yang dihadapi pengguna
        - 📌 Membandingkan performa algoritma Naive Bayes dan SVM
        - 📌 Membuktikan bahwa rating bintang ≠ sentimen teks
        """)
    
    with col2:
        st.markdown("#### 📊 Metodologi")
        st.write("""
        Penelitian ini menggunakan framework **CRISP-DM**:
        
        1. **Business Understanding** - Identifikasi masalah
        2. **Data Understanding** - Eksplorasi dataset
        3. **Data Preparation** - Preprocessing & labeling
        4. **Modeling** - Training Naive Bayes & SVM
        5. **Evaluation** - Perbandingan performa model
        6. **Deployment** - Analisis hasil & rekomendasi
        """)
    
    st.markdown("---")
    
    st.markdown("#### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload file CSV dataset Spotify reviews",
        type=['csv'],
        help="Upload file reviews.csv dari Kaggle: Spotify App Reviews 2022"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Load & Process Data"):
            with st.spinner("Loading data..."):
                df_raw = pd.read_csv(uploaded_file)
                df_raw.columns = df_raw.columns.str.lower()
                
                st.session_state.df_raw = df_raw
                st.session_state.data_loaded = True
                
                st.success(f"✅ Dataset berhasil dimuat: {len(df_raw):,} reviews")
                st.balloons()
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Silakan upload dataset untuk melanjutkan analisis")

# ============================================================================
# DATA UNDERSTANDING PAGE
# ============================================================================

elif page == "📊 Data Understanding":
    st.markdown('<p class="main-header">📊 Data Understanding</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.error("❌ Data belum dimuat. Silakan upload dataset di halaman Home.")
    else:
        df_raw = st.session_state.df_raw
        
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(df_raw):,}")
        col2.metric("Kolom", len(df_raw.columns))
        col3.metric("Missing Values", df_raw.isnull().sum().sum())
        col4.metric("Duplicates", df_raw.duplicated().sum())
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📄 Sample Data", "📊 Rating Distribution", "🔍 Preprocessing"])
        
        with tab1:
            st.markdown("#### Sample Reviews")
            st.dataframe(df_raw.head(10), use_container_width=True)
        
        with tab2:
            st.markdown("#### Rating Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            df_raw['rating'].value_counts().sort_index().plot(kind='bar', color='#1DB954', ax=ax)
            ax.set_title('Distribution of Ratings', fontsize=14, fontweight='bold')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        with tab3:
            st.markdown("#### Data Preprocessing & Labeling")
            
            with st.spinner("Processing data..."):
                # Filter English
                df_raw['is_english'] = df_raw['review'].apply(is_english)
                df = df_raw[df_raw['is_english']].copy().drop('is_english', axis=1)
                
                st.info(f"✅ English reviews filtered: {len(df):,} / {len(df_raw):,}")
                
                # Labeling
                vader = SentimentIntensityAnalyzer()
                df_clean = df[['review', 'rating']].copy()
                df_clean = df_clean.dropna()
                df_clean = df_clean[df_clean['review'].str.strip() != '']
                
                df_clean[['sentiment', 'confidence']] = df_clean['review'].apply(
                    lambda x: pd.Series(label_binary_with_confidence(x, vader))
                )
                
                df_clean = df_clean[df_clean['sentiment'].notna()].copy()
                df_clean['sentiment'] = df_clean['sentiment'].astype(int)
                
                df_train = df_clean[df_clean['confidence'] >= 0.5].copy()
                
                # Save to session state
                st.session_state.df_clean = df_clean
                st.session_state.df_train = df_train
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("After Labeling", f"{len(df_clean):,}")
                    st.metric("High Confidence (≥0.5)", f"{len(df_train):,}")
                
                with col2:
                    sentiment_dist = df_train['sentiment'].value_counts().sort_index()
                    st.write("**Sentiment Distribution:**")
                    st.write(f"- Negative: {sentiment_dist.get(0, 0):,} ({sentiment_dist.get(0, 0)/len(df_train)*100:.1f}%)")
                    st.write(f"- Positive: {sentiment_dist.get(1, 0):,} ({sentiment_dist.get(1, 0)/len(df_train)*100:.1f}%)")
                
                st.success("✅ Data preprocessing completed!")

# ============================================================================
# MODELING PAGE
# ============================================================================

elif page == "🔬 Modeling":
    st.markdown('<p class="main-header">🔬 Modeling & Training</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or 'df_train' not in st.session_state:
        st.error("❌ Data belum diproses. Silakan lakukan preprocessing di halaman Data Understanding.")
    else:
        st.markdown("### 🤖 Train Models")
        
        if st.button("🚀 Start Training", type="primary"):
            df_train = st.session_state.df_train
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Preprocessing
            status_text.text("⏳ Preprocessing text...")
            progress_bar.progress(20)
            
            df_train['review_processed'] = df_train['review'].apply(preprocess_text)
            df_train = df_train[df_train['review_processed'].str.len() > 0]
            
            # TF-IDF
            status_text.text("⏳ Extracting features with TF-IDF...")
            progress_bar.progress(40)
            
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                sublinear_tf=True
            )
            
            X = tfidf.fit_transform(df_train['review_processed'])
            y = df_train['sentiment']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Naive Bayes
            status_text.text("⏳ Training Naive Bayes...")
            progress_bar.progress(60)
            
            param_grid_nb = {'alpha': [0.1, 0.5, 1.0, 2.0]}
            grid_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=3, scoring='accuracy', n_jobs=-1)
            grid_nb.fit(X_train, y_train)
            nb_model = grid_nb.best_estimator_
            y_pred_nb = nb_model.predict(X_test)
            
            # Train SVM
            status_text.text("⏳ Training SVM...")
            progress_bar.progress(80)
            
            param_grid_svm = {'C': [0.5, 1.0, 2.0, 5.0]}
            grid_svm = GridSearchCV(
                SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),
                param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_svm.fit(X_train, y_train)
            svm_model = grid_svm.best_estimator_
            y_pred_svm = svm_model.predict(X_test)
            
            progress_bar.progress(100)
            status_text.text("✅ Training completed!")
            
            # Save to session state
            st.session_state.models_trained = True
            st.session_state.nb_model = nb_model
            st.session_state.svm_model = svm_model
            st.session_state.tfidf = tfidf
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred_nb = y_pred_nb
            st.session_state.y_pred_svm = y_pred_svm
            st.session_state.grid_nb = grid_nb
            st.session_state.grid_svm = grid_svm
            
            st.success("🎉 Models successfully trained!")
            st.balloons()
        
        if st.session_state.models_trained:
            st.markdown("---")
            st.markdown("### ⚙️ Model Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔵 Naive Bayes")
                st.code(f"""
Best Parameters:
- Alpha: {st.session_state.grid_nb.best_params_['alpha']}
- CV Score: {st.session_state.grid_nb.best_score_:.4f}
                """)
            
            with col2:
                st.markdown("#### 🔴 SVM")
                st.code(f"""
Best Parameters:
- C: {st.session_state.grid_svm.best_params_['C']}
- Kernel: linear
- CV Score: {st.session_state.grid_svm.best_score_:.4f}
                """)

# ============================================================================
# RESULTS PAGE
# ============================================================================

elif page == "📈 Results":
    st.markdown('<p class="main-header">📈 Results & Evaluation</p>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.error("❌ Models belum ditraining. Silakan train models di halaman Modeling.")
    else:
        y_test = st.session_state.y_test
        y_pred_nb = st.session_state.y_pred_nb
        y_pred_svm = st.session_state.y_pred_svm
        
        sentiment_labels = {0: 'Negative', 1: 'Positive'}
        
        # Calculate metrics
        def get_metrics(y_true, y_pred):
            return {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        
        nb_scores = get_metrics(y_test, y_pred_nb)
        svm_scores = get_metrics(y_test, y_pred_svm)
        
        # Determine best model
        best_model_name = 'Naive Bayes' if nb_scores['Accuracy'] > svm_scores['Accuracy'] else 'SVM'
        y_pred_best = y_pred_nb if best_model_name == 'Naive Bayes' else y_pred_svm
        
        st.markdown("### 🏆 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", best_model_name)
        col2.metric("Accuracy", f"{max(nb_scores['Accuracy'], svm_scores['Accuracy'])*100:.2f}%")
        col3.metric("Precision", f"{max(nb_scores['Precision'], svm_scores['Precision']):.4f}")
        col4.metric("F1-Score", f"{max(nb_scores['F1-Score'], svm_scores['F1-Score']):.4f}")
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Comparison", "🔵 Naive Bayes", "🔴 SVM"])
        
        with tab1:
            st.markdown("#### Model Comparison")
            
            comparison_df = pd.DataFrame({'Naive Bayes': nb_scores, 'SVM': svm_scores})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(comparison_df.style.highlight_max(axis=1, color='lightgreen'), use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                comparison_df.T.plot(kind='bar', ax=ax, colormap='viridis')
                ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.legend(loc='lower right')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
        
        with tab2:
            st.markdown("#### Naive Bayes - Classification Report")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Metrics:**")
                for metric, value in nb_scores.items():
                    st.metric(metric, f"{value:.4f}")
            
            with col2:
                report = classification_report(y_test, y_pred_nb, 
                                              target_names=['Negative', 'Positive'],
                                              output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        
        with tab3:
            st.markdown("#### SVM - Classification Report")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Metrics:**")
                for metric, value in svm_scores.items():
                    st.metric(metric, f"{value:.4f}")
            
            with col2:
                report = classification_report(y_test, y_pred_svm, 
                                              target_names=['Negative', 'Positive'],
                                              output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Confusion Matrix")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cm_nb = confusion_matrix(y_test, y_pred_nb)
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[0].set_title('Naive Bayes', fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        cm_svm = confusion_matrix(y_test, y_pred_svm)
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[1].set_title('SVM', fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# INSIGHTS PAGE
# ============================================================================

elif page == "💡 Insights":
    st.markdown('<p class="main-header">💡 Insights & Recommendations</p>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.error("❌ Models belum ditraining. Silakan train models terlebih dahulu.")
    else:
        df_clean = st.session_state.df_clean
        best_model = st.session_state.nb_model if st.session_state.nb_model else st.session_state.svm_model
        tfidf = st.session_state.tfidf
        
        with st.spinner("Analyzing all reviews..."):
            df_clean['review_processed'] = df_clean['review'].apply(preprocess_text)
            df_clean = df_clean[df_clean['review_processed'].str.len() > 0]
            
            X_all = tfidf.transform(df_clean['review_processed'])
            df_clean['predicted_sentiment'] = best_model.predict(X_all)
            
            st.session_state.df_analyzed = df_clean
        
        st.markdown("### 📊 Sentiment Distribution")
        
        sentiment_counts = df_clean['predicted_sentiment'].value_counts().sort_index()
        sentiment_labels = {0: 'Negative', 1: 'Positive'}
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", f"{len(df_clean):,}")
        col2.metric("Positive", f"{sentiment_counts.get(1, 0):,} ({sentiment_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
        col3.metric("Negative", f"{sentiment_counts.get(0, 0):,} ({sentiment_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FF6B6B', '#4ECDC4']
        sentiment_counts.plot(kind='bar', color=colors, ax=ax)
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### ⚠️ Rating-Text Mismatch Analysis")
        
        def categorize_rating_binary(rating):
            return 1 if rating >= 4 else 0
        
        df_clean['sentiment_from_rating'] = df_clean['rating'].apply(categorize_rating_binary)
        df_clean['is_mismatch'] = df_clean['predicted_sentiment'] != df_clean['sentiment_from_rating']
        mismatch_pct = (df_clean['is_mismatch'].sum() / len(df_clean)) * 100
        
        st.error(f"🚨 **{mismatch_pct:.1f}%** reviews show rating-text mismatch!")
        
        st.info(f"""
        **KEY INSIGHT:** 
        
        Rating bintang saja **TIDAK CUKUP** untuk mengukur kepuasan pengguna!
        
        {mismatch_pct:.1f}% dari ulasan menunjukkan bahwa rating bintang tidak sesuai dengan 
        sentimen sebenarnya yang tertulis dalam teks ulasan.
        """)
        
        if df_clean['is_mismatch'].sum() > 0:
            st.markdown("#### 📝 Contoh Mismatch")
            mismatch_samples = df_clean[df_clean['is_mismatch']].head(3)
            
            for idx, row in mismatch_samples.iterrows():
                with st.expander(f"Mismatch #{idx}: {row['rating']}⭐ → {sentiment_labels[row['predicted_sentiment']]}"):
                    st.write(f"**Review:** {row['review']}")
        
        st.markdown("---")
        st.markdown("### ☁️ Word Clouds")
        
        col1, col2 = st.columns(2)
        
        def get_meaningful_words(reviews_series, is_negative=False):
            text = " ".join(reviews_series.astype(str)).lower()
            words = text.split()
            
            positive_seeds = {'love', 'great', 'amazing', 'nice', 'cool', 'excellent',
                             'perfect', 'enjoy', 'good', 'best', 'awesome', 'wonderful'}
            negative_seeds = {'crash', 'bug', 'slow', 'error', 'bad', 'hate', 'worst',
                             'lag', 'freeze', 'ads', 'annoying', 'disappointed', 'terrible'}
            
            seeds = negative_seeds if is_negative else positive_seeds
            filtered = [w for w in words if w in seeds]
            return " ".join(filtered)
        
        with col1:
            st.markdown("#### ✅ Positive Reviews")
            reviews_pos = df_clean[df_clean['predicted_sentiment'] == 1]
            if len(reviews_pos) > 0:
                text = get_meaningful_words(reviews_pos['review'], is_negative=False)
                if text and len(text.split()) >= 10:
                    wordcloud = WordCloud(width=400, height=300, background_color='white',
                                         colormap='Greens', max_words=25).generate(text)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
        
        with col2:
            st.markdown("#### ❌ Negative Reviews")
            reviews_neg = df_clean[df_clean['predicted_sentiment'] == 0]
            if len(reviews_neg) > 0:
                text = get_meaningful_words(reviews_neg['review'], is_negative=True)
                if text and len(text.split()) >= 10:
                    wordcloud = WordCloud(width=400, height=300, background_color='white',
                                         colormap='Reds', max_words=25).generate(text)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 🎯 Dominant Issues Analysis")
        
        df_negative = df_clean[df_clean['predicted_sentiment'] == 0]
        
        if len(df_negative) > 0:
            negative_text = " ".join(df_negative['review_processed'])
            negative_words = negative_text.split()
            
            issue_categories = {
                'App Crashes/Freezes': ['crash', 'freez', 'shut', 'close', 'hang', 'stop'],
                'Bugs & Errors': ['bug', 'error', 'glitch', 'problem', 'issu', 'broken'],
                'Performance Issues': ['slow', 'lag', 'buffer', 'load', 'delay', 'wait'],
                'Advertisement Problems': ['ad', 'advertis', 'commerci', 'promo'],
                'Connection Issues': ['connect', 'internet', 'network', 'offlin', 'disconn'],
                'User Dissatisfaction': ['bad', 'hate', 'annoy', 'disappoint', 'frustrat', 'terribl']
            }
            
            category_counts = {}
            for category, keywords in issue_categories.items():
                count = sum(word in keywords for word in negative_words)
                category_counts[category] = count
            
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            
            st.markdown("#### 🔝 Top 3 Issues")
            
            for i, (category, count) in enumerate(sorted_categories[:3], 1):
                pct = (count / len(negative_words)) * 100 if len(negative_words) > 0 else 0
                st.markdown(f"**#{i}. {category}**")
                st.progress(pct/10)
                st.write(f"Mentions: {count} ({pct:.1f}% of negative keywords)")
                st.write("")
            
            st.markdown("---")
            st.markdown("### 📋 Recommendations for Spotify Developers")
            
            recommendations = {
                'App Crashes/Freezes': [
                    "Improve app stability and memory management",
                    "Implement better crash reporting and recovery",
                    "Optimize resource usage to prevent freezes"
                ],
                'Bugs & Errors': [
                    "Enhance QA testing before releases",
                    "Prioritize bug fixes based on user reports",
                    "Implement automated error detection"
                ],
                'Performance Issues': [
                    "Optimize streaming algorithms and caching",
                    "Reduce loading times through better data compression",
                    "Improve performance on low-end devices"
                ],
                'Advertisement Problems': [
                    "Reduce ad frequency in free tier",
                    "Make premium subscription more appealing",
                    "Improve ad relevance and quality"
                ],
                'Connection Issues': [
                    "Enhance offline mode functionality",
                    "Better handling of network interruptions",
                    "Optimize data usage for mobile networks"
                ],
                'User Dissatisfaction': [
                    "Improve customer support responsiveness",
                    "Implement user feedback loops",
                    "Enhance overall user experience"
                ]
            }
            
            for i, (category, _) in enumerate(sorted_categories[:3], 1):
                if category in recommendations:
                    with st.expander(f"💡 Recommendations for: {category}"):
                        for rec in recommendations[category]:
                            st.write(f"• {rec}")

# ============================================================================
# TEST MODEL PAGE
# ============================================================================

elif page == "🧪 Test Model":
    st.markdown('<p class="main-header">🧪 Test Model - Try Your Own Review</p>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.error("❌ Models belum ditraining. Silakan train models terlebih dahulu.")
    else:
        st.markdown("### ✍️ Input Custom Review")
        
        sample_reviews = {
            "Positive Example": "I love this app! The music quality is amazing and the interface is perfect. Best streaming service ever!",
            "Negative Example": "Terrible app! It keeps crashing and freezing. So many bugs and errors. Very disappointed!",
            "Mixed Example": "The app is okay but it has some issues with loading times and occasional crashes."
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_review = st.text_area(
                "Type or paste a review here:",
                height=150,
                placeholder="Example: I love Spotify! Great app with amazing features..."
            )
        
        with col2:
            st.markdown("**Quick Test:**")
            for label, sample in sample_reviews.items():
                if st.button(label):
                    user_review = sample
                    st.session_state.test_review = sample
        
        if 'test_review' in st.session_state:
            user_review = st.session_state.test_review
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if user_review.strip() == "":
                st.warning("⚠️ Please enter a review text!")
            else:
                with st.spinner("Analyzing..."):
                    # Preprocess
                    processed_review = preprocess_text(user_review)
                    
                    if processed_review.strip() == "":
                        st.error("❌ Review too short or contains no meaningful words after preprocessing.")
                    else:
                        # Get models
                        nb_model = st.session_state.nb_model
                        svm_model = st.session_state.svm_model
                        tfidf = st.session_state.tfidf
                        
                        # Transform
                        X_user = tfidf.transform([processed_review])
                        
                        # Predict
                        pred_nb = nb_model.predict(X_user)[0]
                        pred_svm = svm_model.predict(X_user)[0]
                        
                        # Probabilities
                        prob_nb = nb_model.predict_proba(X_user)[0]
                        prob_svm = svm_model.predict_proba(X_user)[0]
                        
                        sentiment_labels = {0: 'Negative', 1: 'Positive'}
                        
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🔵 Naive Bayes Prediction")
                            sentiment_nb = sentiment_labels[pred_nb]
                            color_nb = "green" if pred_nb == 1 else "red"
                            st.markdown(f"<h2 style='color: {color_nb};'>{sentiment_nb}</h2>", unsafe_allow_html=True)
                            
                            st.write("**Confidence:**")
                            st.progress(float(prob_nb[pred_nb]))
                            st.write(f"Negative: {prob_nb[0]*100:.1f}% | Positive: {prob_nb[1]*100:.1f}%")
                        
                        with col2:
                            st.markdown("#### 🔴 SVM Prediction")
                            sentiment_svm = sentiment_labels[pred_svm]
                            color_svm = "green" if pred_svm == 1 else "red"
                            st.markdown(f"<h2 style='color: {color_svm};'>{sentiment_svm}</h2>", unsafe_allow_html=True)
                            
                            st.write("**Confidence:**")
                            st.progress(float(prob_svm[pred_svm]))
                            st.write(f"Negative: {prob_svm[0]*100:.1f}% | Positive: {prob_svm[1]*100:.1f}%")
                        
                        st.markdown("---")
                        
                        if pred_nb == pred_svm:
                            st.success(f"✅ Both models agree: **{sentiment_labels[pred_nb]}**")
                        else:
                            st.warning(f"⚠️ Models disagree! NB: {sentiment_labels[pred_nb]}, SVM: {sentiment_labels[pred_svm]}")
                        
                        with st.expander("🔍 See Processed Text"):
                            st.code(processed_review)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    <p>© 2025 Cici Rahma Angriani</p>
    <p>Politeknik STMI Jakarta</p>
</div>
""", unsafe_allow_html=True)