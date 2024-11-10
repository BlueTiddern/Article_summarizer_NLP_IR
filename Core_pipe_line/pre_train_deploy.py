# Stream lit app for pretrain

import streamlit as st
import pandas as pd
import sqlite3
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from rouge import Rouge
import plotly.express as px

#Simple page configuration 
st.set_page_config(
    page_title="Document Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling and visibility
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 1rem 2rem;
        }
        .summary-container {
            background-color: #2d3436;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 1px solid #636e72;
            color: #dfe6e9;
        }
        .topic-label {
            color: #74b9ff;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .summary-text {
            color: #dfe6e9;
            line-height: 1.6;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #0984e3;
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 600;
        }
        .stButton button:hover {
            background-color: #0866b8;
        }
        /* Input field styling */
        .stTextInput input {
            background-color: #2d3436;
            color: #dfe6e9;
            border: 1px solid #636e72;
        }
        .stTextInput input::placeholder {
            color: #b2bec3;
        }
        /* Main title and description */
        h1, h2, h3, .subtitle {
            color: #dfe6e9 !important;
        }
        p {
            color: #b2bec3;
        }
        /* Tab styling */
        .stTabs [data-baseweb="tab"] {
            color: #dfe6e9 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #0984e3 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header section for my page
st.title("📚 Interactive Document Analysis")
st.markdown('<p class="subtitle">This tool helps you analyze documents and generate topic-wise summaries based on your queries. Enter your question below to get started.</p>', unsafe_allow_html=True)

# loading the resources
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    conn = sqlite3.connect('Modeler_output_topics_gensim.db')
    df_corpus = pd.read_sql("SELECT * FROM topic_data_gensim;", conn)
    conn.close()
    corpus = df_corpus['main_article'].values.astype('U')
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    return model, df_corpus, corpus_embeddings

# Loading models showing the spinner for engagement
with st.spinner('Loading models and data...'):
    model, df_corpus, corpus_embeddings = load_model_and_data()

# custom funtion to retrieve documents
def doc_retriever(question, top_n=5):
    question_embedding = model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, corpus_embeddings)[0]
    top_indices = torch.topk(cosine_scores, k=top_n).indices.tolist()
    top_docs = df_corpus.iloc[top_indices][['highlights', 'topic_label']]
    top_docs['similarity_score'] = cosine_scores[top_indices].tolist()
    return top_docs

# Input section
user_question = st.text_input(
    "Enter your question to get the summaries:",
    placeholder="What is the ques",
    key="question_input"
)

# Add a search button
search_button = st.button("🔍 Search", type="primary")

# check user question validity
if user_question and search_button:
    with st.spinner('Analyzing documents...'):
        # document retrival
        result_transformer = doc_retriever(user_question, top_n=5)
        
        # Grouping with highlights
        grouped_highlights = result_transformer.groupby('topic_label')['highlights'].apply(lambda x: ' '.join(x)).reset_index()
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        grouped_highlights['summary'] = grouped_highlights['highlights'].apply(
            lambda text: summarizer(text, max_length=80, min_length=40, do_sample=False)[0]['summary_text']
        )
        
        # Making the tabs for display
        tab1, tab2 = st.tabs(["📝 Topic Summaries", "📊 Topic Distribution"])
        
        # topic Summaries
        with tab1:
            st.subheader("Topic-wise Summaries")
            for _, row in grouped_highlights.iterrows():
                with st.container(): # container with visual tags
                    st.markdown(f"""
                        <div class="summary-container">
                            <div class="topic-label">🏷️ {row['topic_label']}</div>
                            <div class="summary-text">{row['summary']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        # TOpic distribution handling and plot
        with tab2:
            st.subheader("Topic Distribution")
            topic_counts = result_transformer['topic_label'].value_counts().reset_index()
            topic_counts.columns = ['Topic', 'Count']
            
            fig = px.pie(
                topic_counts,
                values='Count',
                names='Topic',
                title='Distribution for Topics in Retrieved Documents list',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont=dict(color='#2d3436', size=14)  
            )
            fig.update_layout(
                showlegend=False,
                height=500,
                margin=dict(t=50, l=0, r=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#dfe6e9')
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer contents
st.markdown("""
    ---
    <div style="text-align: center; color: #b2bec3;">
        NLP app • Powered by Sentence Transformers and BART
    </div>
""", unsafe_allow_html=True)