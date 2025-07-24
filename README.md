# 🎥 YouTube Video Q&A with RAG

This Streamlit app lets you **chat with the content of any YouTube video** using **Retrieval-Augmented Generation (RAG)**. Just paste a video URL or ID, and ask questions based on its transcript. Powered by **LangChain**, **Groq**, **FAISS**, and **Pinecone**, it delivers fast and accurate answers grounded in the video’s actual content.

---

## 🚀 Features

- 🔍 Extracts transcripts from YouTube videos (if available)
- 🧠 Splits and embeds transcript using HuggingFace embeddings
- 🗂️ Stores vectors locally (FAISS) or in the cloud (Pinecone)
- 🤖 Uses Groq's LLaMA3-70B for fast, context-aware responses
- 🧵 Retrieval chain ensures answers are grounded in transcript
- 🧪 Built with LangChain components for modularity and scalability

---

## 🛠️ Tech Stack

| Component         | Purpose                                |
|------------------|----------------------------------------|
| `Streamlit`      | UI for video input and Q&A             |
| `LangChain`      | RAG pipeline and document handling     |
| `Groq`           | LLM inference with LLaMA3-70B          |
| `FAISS`          | Local vector storage                   |
| `Pinecone`       | Cloud-based vector storage             |
| `YouTubeTranscriptAPI` | Transcript extraction from YouTube |
| `HuggingFaceEmbeddings` | Sentence-transformers for embeddings |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/youtube-rag-chatbot.git
cd youtube-rag-chatbot
pip install -r requirements.txt


env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACEHUB_ACCESS_TOKEN

streamlit run app.py
