# AnsicleXsummarizeR (AXR)

**AnsicleXsummarizeR (AXR)** is an AI-powered web application designed to simplify and enrich the news-reading experience. It integrates question-answering, topic modeling, and contextual summarization to help users retrieve specific answers and understand complex articles more efficiently.

## 🚀 Project Overview

As the volume of digital news grows, readers face difficulty extracting relevant, context-rich information quickly. Traditional search tools often lack semantic depth or topic-wise grouping. AXR addresses this gap by providing:

- **Query-based retrieval of article segments**
- **Topic-wise classification using topic modeling**
- **Contextual summaries tailored to user queries**

## 🧠 Features

- 🔍 **Question Answering**: Find precise answers to user queries from lengthy articles.
- 🧾 **Topic Modeling**: Classifies content into thematic groups using LDA.
- ✂️ **Contextual Summarization**: Generates concise, meaningful summaries using transformer-based models.
- 🌐 **Interactive Web Interface**: Built with Flask for intuitive user interaction.

## 📊 Tech Stack

- **Languages**: Python
- **Libraries**: Hugging Face Transformers, NLTK, Scikit-learn, Gensim
- **Modeling Techniques**: 
  - Latent Dirichlet Allocation (LDA) for topic modeling
  - Transformer-based models for summarization (e.g., BART, T5)
- **Frontend**: HTML/CSS (via Flask templates)

## 🧪 How It Works

1. **User Input**: Upload an article and enter a query.
2. **Processing Pipeline**:
   - Preprocess text (cleaning, tokenization)
   - Identify relevant content sections via QA module
   - Apply LDA for topic grouping
   - Summarize content using transformer-based models
3. **Output**: Topic-wise summaries and direct answers are shown on the web interface.

## ✅ Use Cases

- Journalists and researchers who need to scan long articles quickly.
- Students and readers who prefer simplified, structured content.
- Policy analysts and professionals dealing with domain-heavy text.

## 🔍 Future Enhancements

- Add support for multi-document summarization.
- Integrate speech-to-text for audio input.
- Allow export of summaries to PDF or text formats.

## 🤝 Contributors

- Pavan Yarlagadda  
- Giridhara Srikar Chittem  
- Uday Kiran Chimpiri  
- Manushree Buyya

## 📜 License

This project is for educational and non-commercial use only.

