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
        padding: 1.5rem 0;
        border-bottom: 3px solid #1DB954;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #191414;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1DB954;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #191414;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1DB954;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1DB954;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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

@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()

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

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", width=200)

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Home", "Data Understanding", "Modeling", "Results", "Insights", "Test Model"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Informasi Penelitian")
st.sidebar.markdown("""
**Judul Penelitian:**  
Analisis Sentimen Ulasan Aplikasi Spotify

**Metode yang Digunakan:**
- Naive Bayes (MultinomialNB)
- Support Vector Machine (SVM)

**Dataset:**  
Spotify App Reviews 2022 (Kaggle)

**Peneliti:**  
Cici Rahma Angriani  
NIM: 1323077
""")

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "Home":
    st.markdown('<div class="main-header">ANALISIS SENTIMEN APLIKASI SPOTIFY</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem; font-size: 1.1rem;">Menggunakan Metode Naive Bayes dan Support Vector Machine</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header">Upload Dataset</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Silakan upload file CSV yang berisi data ulasan Spotify. Dataset dapat diunduh dari Kaggle: <b>Spotify App Reviews 2022</b></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV dataset",
        type=['csv'],
        help="Format file: CSV dengan kolom 'review' dan 'rating'"
    )
    
    if uploaded_file is not None:
        if st.button("Load and Process Dataset", type="primary", use_container_width=True):
            with st.spinner("Loading dataset..."):
                df_raw = pd.read_csv(uploaded_file)
                df_raw.columns = df_raw.columns.str.lower()
                
                st.session_state.df_raw = df_raw
                st.session_state.data_loaded = True
            
            st.markdown(f'<div class="success-box">Dataset berhasil dimuat: <b>{len(df_raw):,}</b> total reviews</div>', unsafe_allow_html=True)
            st.balloons()
    
    if not st.session_state.data_loaded:
        st.markdown('<div class="warning-box">Silakan upload dataset untuk memulai analisis</div>', unsafe_allow_html=True)

# ============================================================================
# DATA UNDERSTANDING PAGE
# ============================================================================

elif page == "Data Understanding":
    st.markdown('<div class="main-header">DATA UNDERSTANDING</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div class="error-box">Data belum dimuat. Silakan upload dataset di halaman <b>Home</b>.</div>', unsafe_allow_html=True)
    else:
        df_raw = st.session_state.df_raw
        
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(df_raw):,}")
        col2.metric("Total Kolom", len(df_raw.columns))
        col3.metric("Missing Values", df_raw.isnull().sum().sum())
        col4.metric("Duplicate Rows", df_raw.duplicated().sum())
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["Sample Data", "Rating Distribution", "Data Preprocessing"])
        
        with tab1:
            st.markdown('<div class="subsection-header">Sample Reviews from Dataset</div>', unsafe_allow_html=True)
            st.dataframe(df_raw.head(10), use_container_width=True, height=400)
            
            st.markdown('<div class="subsection-header">Dataset Information</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Column Names:**")
                for col in df_raw.columns:
                    st.write(f"- {col}")
            
            with col2:
                st.markdown("**Data Types:**")
                for col, dtype in df_raw.dtypes.items():
                    st.write(f"- {col}: {dtype}")
        
        with tab2:
            st.markdown('<div class="subsection-header">Rating Distribution Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                rating_counts = df_raw['rating'].value_counts().sort_index()
                bars = ax.bar(rating_counts.index, rating_counts.values, color='#1DB954', 
                            alpha=0.7, edgecolor='black', linewidth=1.2)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax.set_title('Distribution of Star Ratings', fontsize=14, fontweight='bold')
                ax.set_xlabel('Star Rating', fontsize=12, fontweight='bold')
                ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_xticks(range(1, 6))
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Rating Statistics:**")
                for rating in sorted(rating_counts.index, reverse=True):
                    count = rating_counts[rating]
                    percentage = (count / len(df_raw)) * 100
                    st.write(f"**{rating} stars:** {count:,} ({percentage:.1f}%)")
                
                st.markdown("---")
                avg_rating = df_raw['rating'].mean()
                median_rating = df_raw['rating'].median()
                st.metric("Average Rating", f"{avg_rating:.2f}")
                st.metric("Median Rating", f"{median_rating:.0f}")
        
        with tab3:
            st.markdown('<div class="subsection-header">Data Preprocessing and Sentiment Labeling</div>', unsafe_allow_html=True)
            
            if st.button("Start Preprocessing", type="primary"):
                with st.spinner("Processing data..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Filter English
                    status_text.text("Filtering English reviews...")
                    progress_bar.progress(20)
                    df_raw['is_english'] = df_raw['review'].apply(is_english)
                    df = df_raw[df_raw['is_english']].copy().drop('is_english', axis=1)
                    
                    st.markdown(f'<div class="info-box">English reviews filtered: <b>{len(df):,}</b> out of {len(df_raw):,} total reviews</div>', unsafe_allow_html=True)
                    
                    # Labeling
                    status_text.text("Applying sentiment labeling...")
                    progress_bar.progress(50)
                    
                    vader = get_vader()
                    df_clean = df[['review', 'rating']].copy()
                    df_clean = df_clean.dropna()
                    df_clean = df_clean[df_clean['review'].str.strip() != '']
                    
                    df_clean[['sentiment', 'confidence']] = df_clean['review'].apply(
                        lambda x: pd.Series(label_binary_with_confidence(x, vader))
                    )
                    
                    df_clean = df_clean[df_clean['sentiment'].notna()].copy()
                    df_clean['sentiment'] = df_clean['sentiment'].astype(int)
                    
                    status_text.text("Filtering high-confidence samples...")
                    progress_bar.progress(80)
                    
                    df_train = df_clean[df_clean['confidence'] >= 0.5].copy()
                    
                    # Save to session state
                    st.session_state.df = df
                    st.session_state.df_clean = df_clean
                    st.session_state.df_train = df_train
                    
                    progress_bar.progress(100)
                    status_text.text("Preprocessing completed!")
                    
                    st.markdown('<div class="success-box">Data preprocessing and sentiment labeling completed successfully!</div>', unsafe_allow_html=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("After Cleaning", f"{len(df_clean):,}")
                    with col2:
                        st.metric("High Confidence (≥0.5)", f"{len(df_train):,}")
                    with col3:
                        retention_rate = (len(df_train) / len(df_raw) * 100)
                        st.metric("Retention Rate", f"{retention_rate:.1f}%")
                    
                    # Sentiment distribution
                    st.markdown('<div class="subsection-header">Sentiment Distribution (Binary Classification)</div>', unsafe_allow_html=True)
                    
                    sentiment_dist = df_train['sentiment'].value_counts().sort_index()
                    sentiment_labels = {0: 'Negative', 1: 'Positive'}
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#f44336', '#4caf50']
                        labels = [sentiment_labels[0], sentiment_labels[1]]
                        values = [sentiment_dist.get(0, 0), sentiment_dist.get(1, 0)]
                        bars = ax.bar(labels, values, color=colors, alpha=0.8, 
                                    edgecolor='black', linewidth=1.5)
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height):,}',
                                   ha='center', va='bottom', fontsize=11, fontweight='bold')
                        
                        ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
                        ax.set_title('Binary Sentiment Distribution', fontsize=14, fontweight='bold')
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("**Sentiment Breakdown:**")
                        for label in [0, 1]:
                            count = sentiment_dist.get(label, 0)
                            pct = (count / len(df_train) * 100) if len(df_train) > 0 else 0
                            sentiment_name = sentiment_labels[label]
                            st.write(f"**{sentiment_name}:** {count:,} ({pct:.1f}%)")
                        
                        if len(sentiment_dist) == 2:
                            ratio = sentiment_dist.max() / sentiment_dist.min()
                            st.markdown(f"**Imbalance Ratio:** {ratio:.2f}:1")
                            
                            if ratio > 3:
                                st.markdown('<div class="warning-box">Significant class imbalance detected</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="success-box">Classes are reasonably balanced</div>', unsafe_allow_html=True)

# ============================================================================
# MODELING PAGE
# ============================================================================

elif page == "Modeling":
    st.markdown('<div class="main-header">MODEL TRAINING</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or 'df_train' not in st.session_state:
        st.markdown('<div class="error-box">Data belum diproses. Silakan lakukan preprocessing di halaman <b>Data Understanding</b>.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-header">TF-IDF Feature Extraction</div>', unsafe_allow_html=True)
            st.markdown("""
            **Vectorization Parameters:**
            - Maximum Features: 5000
            - N-gram Range: (1, 2) - unigrams and bigrams
            - Minimum Document Frequency: 5
            - Maximum Document Frequency: 0.9
            - Sublinear TF Scaling: True
            """)
        
        with col2:
            st.markdown('<div class="subsection-header">Model Parameters</div>', unsafe_allow_html=True)
            st.markdown("""
            **Naive Bayes Configuration:**
            - Algorithm: MultinomialNB
            - Alpha Tuning: [0.1, 0.5, 1.0, 2.0]
            
            **SVM Configuration:**
            - Kernel: Linear
            - C Parameter: [0.5, 1.0, 2.0, 5.0]
            - Class Weight: Balanced
            """)
        
        st.markdown("---")
        
        if st.button("Start Model Training", type="primary", use_container_width=True):
            df_train = st.session_state.df_train
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Preprocessing
            status_text.text("Preprocessing text data...")
            progress_bar.progress(10)
            
            df_train['review_processed'] = df_train['review'].apply(preprocess_text)
            df_train = df_train[df_train['review_processed'].str.len() > 0]
            
            # TF-IDF
            status_text.text("Extracting features using TF-IDF...")
            progress_bar.progress(25)
            
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                sublinear_tf=True
            )
            
            X = tfidf.fit_transform(df_train['review_processed'])
            y = df_train['sentiment']
            
            st.markdown(f'<div class="info-box">TF-IDF Feature Matrix: <b>{X.shape[0]}</b> samples × <b>{X.shape[1]}</b> features</div>', unsafe_allow_html=True)
            
            # Train-test split
            status_text.text("Splitting data into train and test sets...")
            progress_bar.progress(35)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            col1, col2 = st.columns(2)
            col1.metric("Training Set", f"{X_train.shape[0]:,} samples")
            col2.metric("Testing Set", f"{X_test.shape[0]:,} samples")
            
            # Train Naive Bayes
            st.markdown('<div class="subsection-header">Training Naive Bayes Classifier</div>', unsafe_allow_html=True)
            status_text.text("Running GridSearchCV for Naive Bayes...")
            progress_bar.progress(50)
            
            param_grid_nb = {'alpha': [0.1, 0.5, 1.0, 2.0]}
            grid_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=3, scoring='accuracy', n_jobs=-1)
            grid_nb.fit(X_train, y_train)
            nb_model = grid_nb.best_estimator_
            y_pred_nb = nb_model.predict(X_test)
            
            st.write(f"Best Parameters: alpha = {grid_nb.best_params_['alpha']}")
            st.write(f"Cross-Validation Accuracy: {grid_nb.best_score_:.4f}")
            
            # Train SVM
            st.markdown('<div class="subsection-header">Training Support Vector Machine</div>', unsafe_allow_html=True)
            status_text.text("Running GridSearchCV for SVM...")
            progress_bar.progress(75)
            
            param_grid_svm = {'C': [0.5, 1.0, 2.0, 5.0]}
            grid_svm = GridSearchCV(
                SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),
                param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_svm.fit(X_train, y_train)
            svm_model = grid_svm.best_estimator_
            y_pred_svm = svm_model.predict(X_test)
            
            st.write(f"Best Parameters: C = {grid_svm.best_params_['C']}")
            st.write(f"Cross-Validation Accuracy: {grid_svm.best_score_:.4f}")
            
            progress_bar.progress(100)
            status_text.text("Training completed successfully!")
            
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
            
            st.markdown('<div class="success-box">Both models have been trained and optimized successfully. Proceed to <b>Results</b> page to compare their performance.</div>', unsafe_allow_html=True)
            st.balloons()
        
        if st.session_state.models_trained:
            st.markdown("---")
            st.markdown('<div class="section-header">Training Summary</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Naive Bayes Configuration</div>', unsafe_allow_html=True)
                st.markdown(f"""
                - **Algorithm:** MultinomialNB
                - **Best Alpha:** {st.session_state.grid_nb.best_params_['alpha']}
                - **CV Accuracy:** {st.session_state.grid_nb.best_score_:.4f}
                - **Cross-Validation:** 3-fold
                """)
            
            with col2:
                st.markdown('<div class="subsection-header">SVM Configuration</div>', unsafe_allow_html=True)
                st.markdown(f"""
                - **Kernel:** Linear
                - **Best C:** {st.session_state.grid_svm.best_params_['C']}
                - **CV Accuracy:** {st.session_state.grid_svm.best_score_:.4f}
                - **Class Weight:** Balanced
                """)

# ============================================================================
# RESULTS PAGE
# ============================================================================

elif page == "Results":
    st.markdown('<div class="main-header">RESULTS AND EVALUATION</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.markdown('<div class="error-box">Models belum ditraining. Silakan train models di halaman <b>Modeling</b>.</div>', unsafe_allow_html=True)
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
        
        st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="success-box"><b>Best Performing Model:</b> {best_model_name} with accuracy of {max(nb_scores["Accuracy"], svm_scores["Accuracy"])*100:.2f}%</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", best_model_name)
        col2.metric("Accuracy", f"{max(nb_scores['Accuracy'], svm_scores['Accuracy'])*100:.2f}%")
        col3.metric("Precision", f"{max(nb_scores['Precision'], svm_scores['Precision']):.4f}")
        col4.metric("F1-Score", f"{max(nb_scores['F1-Score'], svm_scores['F1-Score']):.4f}")
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["Model Comparison", "Naive Bayes Details", "SVM Details"])
        
        with tab1:
            st.markdown('<div class="subsection-header">Performance Metrics Comparison</div>', unsafe_allow_html=True)
            
            comparison_df = pd.DataFrame({'Naive Bayes': nb_scores, 'SVM': svm_scores})
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Metrics Table:**")
                comparison_display = comparison_df.copy()
                for col in comparison_display.columns:
                    comparison_display[col] = comparison_display[col].apply(lambda x: f"{x:.4f}")
                st.dataframe(comparison_display, use_container_width=True)
            
            with col2:
                st.markdown("**Visual Comparison:**")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(comparison_df.index))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, comparison_df['Naive Bayes'], width, 
                              label='Naive Bayes', color='#2196f3', alpha=0.8, edgecolor='black')
                bars2 = ax.bar(x + width/2, comparison_df['SVM'], width, 
                              label='SVM', color='#ff9800', alpha=0.8, edgecolor='black')
                
                ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
                ax.set_ylabel('Score', fontsize=12, fontweight='bold')
                ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(comparison_df.index, rotation=0)
                ax.legend(loc='lower right', fontsize=10)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_ylim([0, 1.1])
                
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom', fontsize=9)
                
                st.pyplot(fig)
        
        with tab2:
            st.markdown('<div class="subsection-header">Naive Bayes Classification Report</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Performance Metrics:**")
                for metric, value in nb_scores.items():
                    st.metric(metric, f"{value:.4f}")
            
            with col2:
                st.markdown("**Detailed Report:**")
                report = classification_report(y_test, y_pred_nb, 
                                              target_names=['Negative', 'Positive'],
                                              output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                report_df = report_df.round(4)
                st.dataframe(report_df, use_container_width=True)
        
        with tab3:
            st.markdown('<div class="subsection-header">SVM Classification Report</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Performance Metrics:**")
                for metric, value in svm_scores.items():
                    st.metric(metric, f"{value:.4f}")
            
            with col2:
                st.markdown("**Detailed Report:**")
                report = classification_report(y_test, y_pred_svm, 
                                              target_names=['Negative', 'Positive'],
                                              output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                report_df = report_df.round(4)
                st.dataframe(report_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown('<div class="section-header">Confusion Matrix Analysis</div>', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        cm_nb = confusion_matrix(y_test, y_pred_nb)
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        axes[0].set_title('Naive Bayes Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Actual Label', fontsize=12, fontweight='bold')
        
        cm_svm = confusion_matrix(y_test, y_pred_svm)
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        axes[1].set_title('SVM Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Actual Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Model interpretation
        st.markdown('<div class="subsection-header">Performance Interpretation</div>', unsafe_allow_html=True)
        
        accuracy_diff = abs(nb_scores['Accuracy'] - svm_scores['Accuracy'])
        
        if accuracy_diff < 0.02:
            interpretation = f"Both models show very similar performance with only {accuracy_diff*100:.2f}% difference in accuracy. This suggests that both algorithms are well-suited for this binary sentiment classification task."
        elif best_model_name == 'Naive Bayes':
            interpretation = f"Naive Bayes outperforms SVM by {accuracy_diff*100:.2f}%, likely due to the independence assumption working well with TF-IDF features in this sentiment analysis context."
        else:
            interpretation = f"SVM outperforms Naive Bayes by {accuracy_diff*100:.2f}%, suggesting that the linear decision boundary found by SVM is more effective at separating positive and negative sentiments in the feature space."
        
        st.markdown(f'<div class="info-box">{interpretation}</div>', unsafe_allow_html=True)

# ============================================================================
# INSIGHTS PAGE
# ============================================================================

elif page == "Insights":
    st.markdown('<div class="main-header">INSIGHTS AND RECOMMENDATIONS</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.markdown('<div class="error-box">Models belum ditraining. Silakan train models terlebih dahulu.</div>', unsafe_allow_html=True)
    else:
        df_clean = st.session_state.df_clean
        
        # Determine best model
        nb_scores = {'Accuracy': accuracy_score(st.session_state.y_test, st.session_state.y_pred_nb)}
        svm_scores = {'Accuracy': accuracy_score(st.session_state.y_test, st.session_state.y_pred_svm)}
        best_model_name = 'Naive Bayes' if nb_scores['Accuracy'] > svm_scores['Accuracy'] else 'SVM'
        best_model = st.session_state.nb_model if best_model_name == 'Naive Bayes' else st.session_state.svm_model
        tfidf = st.session_state.tfidf
        
        with st.spinner("Analyzing all reviews..."):
            df_clean['review_processed'] = df_clean['review'].apply(preprocess_text)
            df_clean = df_clean[df_clean['review_processed'].str.len() > 0]
            
            X_all = tfidf.transform(df_clean['review_processed'])
            df_clean['predicted_sentiment'] = best_model.predict(X_all)
            
            st.session_state.df_analyzed = df_clean
        
        st.markdown(f'<div class="success-box">Analysis completed using <b>{best_model_name}</b> model on {len(df_clean):,} reviews</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Overall Sentiment Distribution</div>', unsafe_allow_html=True)
        
        sentiment_counts = df_clean['predicted_sentiment'].value_counts().sort_index()
        sentiment_labels = {0: 'Negative', 1: 'Positive'}
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Analyzed", f"{len(df_clean):,}")
        col2.metric("Positive Reviews", f"{sentiment_counts.get(1, 0):,} ({sentiment_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
        col3.metric("Negative Reviews", f"{sentiment_counts.get(0, 0):,} ({sentiment_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#f44336', '#4caf50']
            labels = [sentiment_labels[0], sentiment_labels[1]]
            sizes = [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                               colors=colors, startangle=90,
                                               textprops={'fontsize': 12, 'fontweight': 'bold'})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
                autotext.set_fontweight('bold')
            
            ax.set_title('Predicted Sentiment Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar([sentiment_labels[0], sentiment_labels[1]], 
                         [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
            ax.set_title('Sentiment Count Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown('<div class="section-header">Rating vs Text Sentiment Mismatch Analysis</div>', unsafe_allow_html=True)
        
        def categorize_rating_binary(rating):
            return 1 if rating >= 4 else 0
        
        df_clean['sentiment_from_rating'] = df_clean['rating'].apply(categorize_rating_binary)
        df_clean['is_mismatch'] = df_clean['predicted_sentiment'] != df_clean['sentiment_from_rating']
        mismatch_count = df_clean['is_mismatch'].sum()
        mismatch_pct = (mismatch_count / len(df_clean)) * 100
        
        st.markdown(f'<div class="warning-box"><b>Critical Finding:</b> {mismatch_pct:.1f}% of reviews show rating-text sentiment mismatch!</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
        <b>Key Insight:</b><br><br>
        Star ratings alone are <b>NOT SUFFICIENT</b> to measure true user satisfaction!<br><br>
        {mismatch_pct:.1f}% of reviews demonstrate that star ratings do not accurately reflect 
        the actual sentiment expressed in the review text. This highlights the importance of 
        natural language processing in understanding customer feedback.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Mismatches", f"{mismatch_count:,}")
            st.metric("Mismatch Rate", f"{mismatch_pct:.1f}%")
        
        with col2:
            st.metric("Matches", f"{len(df_clean) - mismatch_count:,}")
            st.metric("Match Rate", f"{100 - mismatch_pct:.1f}%")
        
        if mismatch_count > 0:
            st.markdown('<div class="subsection-header">Sample Mismatch Cases</div>', unsafe_allow_html=True)
            mismatch_samples = df_clean[df_clean['is_mismatch']].head(5)
            
            for idx, row in mismatch_samples.iterrows():
                pred_label = sentiment_labels[row['predicted_sentiment']]
                rating_label = sentiment_labels[row['sentiment_from_rating']]
                
                with st.expander(f"Rating: {row['rating']} stars ({rating_label}) vs Text: {pred_label}"):
                    st.write("**Review Text:**")
                    st.write(row['review'])
                    st.write(f"**Confidence Score:** {row['confidence']:.2f}")
        
        st.markdown("---")
        st.markdown('<div class="section-header">Word Cloud Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-header">Positive Reviews - Key Words</div>', unsafe_allow_html=True)
            reviews_pos = df_clean[df_clean['predicted_sentiment'] == 1]
            if len(reviews_pos) > 0:
                text = get_meaningful_words(reviews_pos['review'], is_negative=False)
                if text and len(text.split()) >= 10:
                    wordcloud = WordCloud(width=500, height=350, background_color='white',
                                         colormap='Greens', max_words=30, relative_scaling=0.5,
                                         min_word_length=4, collocations=False).generate(text)
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.set_title('Positive Reviews - Dominant Words', fontsize=14, fontweight='bold')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Insufficient data for positive word cloud")
        
        with col2:
            st.markdown('<div class="subsection-header">Negative Reviews - Key Words</div>', unsafe_allow_html=True)
            reviews_neg = df_clean[df_clean['predicted_sentiment'] == 0]
            if len(reviews_neg) > 0:
                text = get_meaningful_words(reviews_neg['review'], is_negative=True)
                if text and len(text.split()) >= 10:
                    wordcloud = WordCloud(width=500, height=350, background_color='white',
                                         colormap='Reds', max_words=30, relative_scaling=0.5,
                                         min_word_length=4, collocations=False).generate(text)
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.set_title('Negative Reviews - Dominant Words', fontsize=14, fontweight='bold')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Insufficient data for negative word cloud")
        
        st.markdown("---")
        st.markdown('<div class="section-header">Dominant Issues in Negative Reviews</div>', unsafe_allow_html=True)
        
        df_negative = df_clean[df_clean['predicted_sentiment'] == 0]
        
        if len(df_negative) > 0:
            st.write(f"Analyzing {len(df_negative):,} negative reviews for common issues...")
            
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
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="subsection-header">Issue Distribution</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                categories = [cat for cat, _ in sorted_categories]
                counts = [count for _, count in sorted_categories]
                
                bars = ax.barh(categories, counts, color='#f44336', alpha=0.7, edgecolor='black')
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 10, bar.get_y() + bar.get_height()/2., f'{int(width):,}',
                           ha='left', va='center', fontsize=10, fontweight='bold')
                
                ax.set_xlabel('Number of Mentions', fontsize=12, fontweight='bold')
                ax.set_title('Issue Categories in Negative Reviews', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Top 3 Issues</div>', unsafe_allow_html=True)
                for i, (category, count) in enumerate(sorted_categories[:3], 1):
                    pct = (count / len(negative_words)) * 100 if len(negative_words) > 0 else 0
                    st.write(f"**{i}. {category}**")
                    st.write(f"Mentions: {count:,} ({pct:.1f}%)")
                    st.progress(min(pct / 10, 1.0))
                    st.write("")
            
            st.markdown("---")
            st.markdown('<div class="section-header">Recommendations for Spotify Developers</div>', unsafe_allow_html=True)
            
            recommendations = {
                'App Crashes/Freezes': [
                    "Improve app stability and memory management",
                    "Implement better crash reporting and recovery mechanisms",
                    "Optimize resource usage to prevent freezes",
                    "Conduct thorough testing on various device configurations"
                ],
                'Bugs & Errors': [
                    "Enhance quality assurance testing before releases",
                    "Prioritize bug fixes based on user impact and frequency",
                    "Implement automated error detection and reporting systems",
                    "Establish a rapid response team for critical bugs"
                ],
                'Performance Issues': [
                    "Optimize streaming algorithms and caching strategies",
                    "Reduce loading times through better data compression",
                    "Improve performance on low-end and older devices",
                    "Implement adaptive quality settings based on network conditions"
                ],
                'Advertisement Problems': [
                    "Reduce advertisement frequency in free tier",
                    "Make premium subscription benefits more compelling",
                    "Improve advertisement relevance and targeting",
                    "Provide skip options for less intrusive experience"
                ],
                'Connection Issues': [
                    "Enhance offline mode functionality and reliability",
                    "Better handling of network interruptions and reconnection",
                    "Optimize data usage for mobile and limited networks",
                    "Implement intelligent pre-caching for frequently played content"
                ],
                'User Dissatisfaction': [
                    "Improve customer support responsiveness and accessibility",
                    "Implement continuous user feedback collection loops",
                    "Enhance overall user experience based on sentiment analysis",
                    "Create transparent communication channels for updates"
                ]
            }
            
            for i, (category, _) in enumerate(sorted_categories[:3], 1):
                if category in recommendations:
                    with st.expander(f"Recommendations for Issue #{i}: {category}"):
                        for rec in recommendations[category]:
                            st.write(f"• {rec}")
        else:
            st.info("No negative reviews found in predictions")
        
        st.markdown("---")
        st.markdown('<div class="section-header">Research Summary and Conclusions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="subsection-header">Quantitative Findings</div>', unsafe_allow_html=True)
            st.write(f"• Total reviews analyzed: {len(df_clean):,}")
            st.write(f"• Positive sentiment: {sentiment_counts.get(1, 0):,} ({sentiment_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
            st.write(f"• Negative sentiment: {sentiment_counts.get(0, 0):,} ({sentiment_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
            st.write(f"• Rating-text mismatch rate: {mismatch_pct:.1f}%")
            st.write(f"• Best performing model: {best_model_name}")
            best_accuracy = max(nb_scores['Accuracy'], svm_scores['Accuracy'])
            st.write(f"• Model accuracy: {best_accuracy*100:.2f}%")
        
        with col2:
            st.markdown('<div class="subsection-header">Key Conclusions</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="success-box">
            <b>Primary Finding:</b><br>
            Star ratings alone are insufficient for measuring true user satisfaction. 
            Text analysis reveals actual sentiment in {mismatch_pct:.1f}% of cases where 
            ratings do not match review content.<br><br>
            
            <b>Implications for Industry:</b><br>
            Companies should implement natural language processing alongside traditional 
            rating systems to gain accurate insights into customer sentiment and satisfaction 
            levels for better product development decisions.
            </b>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TEST MODEL PAGE
# ============================================================================

elif page == "Test Model":
    st.markdown('<div class="main-header">TEST MODEL WITH CUSTOM REVIEW</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.markdown('<div class="error-box">Models belum ditraining. Silakan train models terlebih dahulu.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">Try Your Own Review</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">Enter a custom review text to see how both models classify the sentiment. You can type your own review or use one of the sample reviews provided.</div>', unsafe_allow_html=True)
        
        sample_reviews = {
            "Positive Example": "I love this app! The music quality is amazing and the interface is perfect. Best streaming service ever!",
            "Negative Example": "Terrible app! It keeps crashing and freezing. So many bugs and errors. Very disappointed!",
            "Mixed Example": "The app is okay but it has some issues with loading times and occasional crashes."
        }
        
        col1, col2 = st.columns([2, 1])
        
        st.markdown('<div class="subsection-header">Input Review Text</div>', unsafe_allow_html=True)
        user_review = st.text_area(
                "Type or paste your review here:",
                height=150,
                placeholder="Example: I love Spotify! Great app with amazing features...",
                label_visibility="collapsed"
        )
        
        if st.button("Analyze Sentiment", type="primary", use_container_width=True):
            if user_review.strip() == "":
                st.markdown('<div class="warning-box">Please enter a review text!</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Analyzing sentiment..."):
                    # Preprocess
                    processed_review = preprocess_text(user_review)
                    
                    if processed_review.strip() == "":
                        st.markdown('<div class="error-box">Review too short or contains no meaningful words after preprocessing.</div>', unsafe_allow_html=True)
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
                        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="subsection-header">Naive Bayes Prediction</div>', unsafe_allow_html=True)
                            sentiment_nb = sentiment_labels[pred_nb]
                            color_nb = "green" if pred_nb == 1 else "red"
                            st.markdown(f"<h2 style='color: {color_nb}; text-align: center;'>{sentiment_nb}</h2>", unsafe_allow_html=True)
                            
                            st.markdown("**Confidence Distribution:**")
                            st.progress(float(prob_nb[pred_nb]))
                            st.write(f"Negative: {prob_nb[0]*100:.1f}% | Positive: {prob_nb[1]*100:.1f}%")
                            
                            st.metric("Prediction Confidence", f"{prob_nb[pred_nb]*100:.1f}%")
                        
                        with col2:
                            st.markdown('<div class="subsection-header">SVM Prediction</div>', unsafe_allow_html=True)
                            sentiment_svm = sentiment_labels[pred_svm]
                            color_svm = "green" if pred_svm == 1 else "red"
                            st.markdown(f"<h2 style='color: {color_svm}; text-align: center;'>{sentiment_svm}</h2>", unsafe_allow_html=True)
                            
                            st.markdown("**Confidence Distribution:**")
                            st.progress(float(prob_svm[pred_svm]))
                            st.write(f"Negative: {prob_svm[0]*100:.1f}% | Positive: {prob_svm[1]*100:.1f}%")
                            
                            st.metric("Prediction Confidence", f"{prob_svm[pred_svm]*100:.1f}%")
                        
                        st.markdown("---")
                        
                        if pred_nb == pred_svm:
                            st.markdown(f'<div class="success-box"><b>Model Agreement:</b> Both models agree on <b>{sentiment_labels[pred_nb]}</b> sentiment!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="warning-box"><b>Model Disagreement:</b> Naive Bayes predicts <b>{sentiment_labels[pred_nb]}</b> while SVM predicts <b>{sentiment_labels[pred_svm]}</b></div>', unsafe_allow_html=True)
                        
                        # Visualization
                        st.markdown('<div class="subsection-header">Prediction Confidence Comparison</div>', unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        x = np.arange(2)
                        width = 0.35
                        
                        bars1 = ax.bar(x - width/2, [prob_nb[0], prob_nb[1]], width,
                                      label='Naive Bayes', color=['#f44336', '#4caf50'], alpha=0.8, edgecolor='black')
                        bars2 = ax.bar(x + width/2, [prob_svm[0], prob_svm[1]], width,
                                      label='SVM', color=['#ff9800', '#2196f3'], alpha=0.8, edgecolor='black')
                        
                        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
                        ax.set_title('Sentiment Prediction Probabilities', fontsize=14, fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(['Negative', 'Positive'])
                        ax.legend()
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        ax.set_ylim([0, 1.1])
                        
                        for bars in [bars1, bars2]:
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.3f}',
                                       ha='center', va='bottom', fontsize=9)
                        
                        st.pyplot(fig)
                        
                        with st.expander("View Preprocessed Text"):
                            st.code(processed_review)
                        
                        with st.expander("View Original Review"):
                            st.write(user_review)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p style='margin-bottom: 0.5rem; font-size: 0.9rem;'><b>Spotify Sentiment Analysis</b></p>
    <p style='margin-bottom: 0.5rem; font-size: 0.85rem;'>Cici Rahma Angriani</p>
    <p style='margin-bottom: 0.5rem; font-size: 0.85rem;'>NIM: 1323077</p>
    <p style='font-size: 0.8rem; color: #999;'>Politeknik STMI Jakarta</p>
</div>
""", unsafe_allow_html=True)