# streamlit_app for stat model

import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from textblob import TextBlob
from nltk.corpus import wordnet as wn
import gensim.downloader as api
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import numpy as np
import plotly.express as px

# Page set up
st.set_page_config(
    page_title="AXR Document Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for stream streamlit app
st.markdown("""
    <style>
        /* Main container and background */
        .main {
            background-color: #1a1b1e;
            color: #e0e0e0;
        }
        
        /* Search container */
        .search-container {
            background: linear-gradient(180deg, rgba(30, 60, 114, 0.2) 0%, rgba(30, 60, 114, 0.1) 100%);
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        /* Results container */
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Text styling */
        .highlight-text {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #e0e0e0;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            margin: 0.5rem 0;
        }
        
        /* Topic label */
        .topic-label {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            background: #1e3c72;
            border-radius: 15px;
            font-size: 0.9rem;
            margin: 0.5rem 0;
        }
        
        /* Metrics styling */
        .metric-container {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem;
        }
        
        /* Search input */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 8px;
            padding: 0.8rem;
        }
        
        /* Button styling */
        .stButton > button {
            background: #1e3c72;
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 8px;
            width: 100%;
        }
        
        /* Tab styling */
        .stTabs > div > div > div {
            background-color: transparent;
            color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# Downloading required resources
@st.cache_resource
def load_resources():
    """Initialize and load required NLP resources"""
    nltk.download('wordnet')
    return api.load("glove-wiki-gigaword-100")

# Word_vect loading
word_vectors = load_resources()

# Database connection and data loading
@st.cache_resource
def load_database():
    """Connect to SQLite database and load corpus data"""
    conn = sqlite3.connect('Modeler_output_topics_gensim.db')
    df = pd.read_sql("SELECT * FROM topic_data_gensim;", conn)
    conn.close()
    return df

# Load the corpus
df_corpus = load_database()

# Initialize TF-IDF vectorizer
@st.cache_resource
def initialize_tfidf():
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=2,
        norm='l2',
        use_idf=True
    )
    corpus = df_corpus['main_article'].values.astype('U')
    tf_idf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tf_idf_matrix

# Initialize TF-IDF matrices
tfidf_vectorizer, tf_idf_m = initialize_tfidf()

# Document retrieval function
def doc_retrieve(question, top_n=5):
    """
    Below is the description for the fucntion
    Retrieve most relevant documents for given query
    Args:
        question (str): User query
        top_n (int): Number of documents to retrieve
    Returns:
        pd.DataFrame: Top matching documents with similarity scores
    """
    u_ques_vec = tfidf_vectorizer.transform([question])
    cosine_similarities = linear_kernel(u_ques_vec, tf_idf_m).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_docs = df_corpus.iloc[top_indices][['highlights', 'topic_label']]
    top_docs['similarity_score'] = cosine_similarities[top_indices]
    return top_docs

# Spelling correction function
def correct_spelling(text):
    """Correct spelling errors in text"""
    blob = TextBlob(text)
    return str(blob.correct())

# Query expansion function
def expand_query(query):
    """
    Below is the description for the fucntion
    Expand query with synonyms and corrections
    Args:
        query (str): Original query
    Returns:
        str: Expanded query
    """
    corrected_q = correct_spelling(query)
    expanded_query = corrected_q.split()
    for word in query.split():
        synonyms = wn.synsets(word)
        for syn in synonyms[:2]:
            for lemma in syn.lemmas():
                expanded_query.append(lemma.name().replace('_', ' '))
    return ' '.join(set(expanded_query))

# LSA summarization function
def lsa_summary_gensim(df, topic_column='topic_label', text_column='highlights', num_phrases=100):
    """
    Below is the description for the fucntion
    Generate LSA-based summaries for topics
    Args:
        df (pd.DataFrame): Input dataframe
        topic_column (str): Column name for topics
        text_column (str): Column name for text content
        num_phrases (int): Number of phrases to include in summary
    Returns:
        pd.DataFrame: Topic summaries
    """
    topic_summaries = {}
    grouped = df.groupby(topic_column)
    
    for topic, group in grouped:
        combined_text = ' '.join(group[text_column].dropna().values)
        if not combined_text.strip():
            topic_summaries[topic] = "No content to summarize."
            continue
            
        try:
            tfidf_vectorizer_LSA = TfidfVectorizer(
                sublinear_tf=True,
                max_features=1500,
                ngram_range=(1, 2),
                stop_words='english'
            )
            tfidf_matrix = tfidf_vectorizer_LSA.fit_transform([combined_text])
            tfidf_matrix = normalize(tfidf_matrix, norm='l2')
            
            if tfidf_matrix.sum() == 0:
                topic_summaries[topic] = "No significant content to summarize."
                continue
                
            svd = TruncatedSVD(n_components=1, tol=1e-4, n_iter=10)
            svd.fit(tfidf_matrix)
            
        except ValueError as e:
            topic_summaries[topic] = "Error processing text."
            continue
            
        word_importances = svd.components_[0]
        feature_names = tfidf_vectorizer_LSA.get_feature_names_out()
        word_importance_pairs = [(feature_names[i], word_importances[i]) for i in range(len(word_importances))]
        word_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_words = [pair[0] for pair in word_importance_pairs[:num_phrases]]
        top_words_with_embeddings = [word for word in top_words if word in word_vectors]
        
        if top_words_with_embeddings:
            summary = ' '.join(top_words_with_embeddings)
        else:
            summary = "No valid summary generated."
            
        topic_summaries[topic] = summary
        
    return pd.DataFrame(list(topic_summaries.items()), columns=[topic_column, 'summary'])

# Streamlit main UI component
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.title("📚 AXR Document Analytics")
st.markdown("Semantic retrival and summarization")
st.markdown('</div>', unsafe_allow_html=True)

# Creating a simple Search interface
query = st.text_input("🔍 Enter your search query:",  # helping text for the user
                     placeholder="Type your query here...",
                     help="Enter keywords or phrases to search through the document database")

# Search button
search_clicked = st.button("Search Documents", use_container_width=True)

if search_clicked and query:
    with st.spinner("Processing your query..."):
        # Query expansion
        expanded_query = expand_query(query)
        
        # Document retrieval
        results = doc_retrieve(expanded_query)
        
        # Topic summarization
        summary_df = lsa_summary_gensim(results)
    
    # Simple statistical metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents Found", len(results))
    with col2:
        st.metric("Topics Covered", len(results['topic_label'].unique()))
    with col3:
        st.metric("Avg. Similarity", f"{results['similarity_score'].mean():.2f}")
    
    # Tabs on the page with icons
    tab1, tab2, tab3 = st.tabs(["📝 Documents", "📊 Topics", "📈 Analysis"])
    
    with tab1:
        # These are the retrievers output
        for idx, row in results.iterrows():
            st.markdown(f"""
                <div class="result-card">
                    <div class="topic-label">{row['topic_label']}</div>
                    <div class="highlight-text">{row['highlights']}</div>
                    <div style="text-align: right; color: #888;">
                        Similarity: {row['similarity_score']:.2%}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # This tab will display the topic vice summaries
        for idx, row in summary_df.iterrows():
            st.markdown(f"""
                <div class="result-card">
                    <div class="topic-label">{row['topic_label']}</div>
                    <div class="highlight-text">{row['summary']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        # This tab is for topic distribution visualization
        topic_dist = results['topic_label'].value_counts()
        fig = px.pie(
            values=topic_dist.values,
            names=topic_dist.index,
            title='Topic Distribution',
            hole=0.4,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Similarity score distribution using histogram
        fig = px.histogram(
            results,
            x='similarity_score',
            nbins=20,
            title='Document Similarity Distribution',
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

# Generic footer to display the base on the page
st.markdown("""
    <div style='text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(30, 60, 114, 0.1); border-radius: 10px;'>
        <p>AXR • Powered by Advanced NLP by py</p>
    </div>
""", unsafe_allow_html=True)