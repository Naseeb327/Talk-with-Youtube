import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv

load_dotenv()

# Load API keys
groq_api_key = os.environ.get('GROQ_API_KEY')
pinecone_api = os.environ.get("PINECONE_API_KEY")

def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return ID if already provided"""
    # Clean the input first
    url_or_id = url_or_id.strip()
    
    # Handle nested URLs (like the error case)
    if url_or_id.count("youtube.com") > 1:
        # Find the last occurrence which should be the actual video URL
        parts = url_or_id.split("youtube.com")
        url_or_id = "youtube.com" + parts[-1]
    
    # Handle different YouTube URL formats
    if "youtube.com/watch?v=" in url_or_id:
        video_id = url_or_id.split("watch?v=")[1].split("&")[0].split("#")[0]
        return video_id
    elif "youtu.be/" in url_or_id:
        video_id = url_or_id.split("youtu.be/")[1].split("?")[0].split("#")[0]
        return video_id
    elif "youtube.com/live/" in url_or_id:
        # Handle YouTube Live URLs
        video_id = url_or_id.split("youtube.com/live/")[1].split("?")[0].split("#")[0]
        return video_id
    elif "youtube.com/embed/" in url_or_id:
        # Handle embedded URLs
        video_id = url_or_id.split("youtube.com/embed/")[1].split("?")[0].split("#")[0]
        return video_id
    else:
        # Assume it's already a video ID - validate it's roughly the right format
        cleaned_id = url_or_id.split("?")[0].split("#")[0]
        if len(cleaned_id) == 11 and cleaned_id.replace("-", "").replace("_", "").isalnum():
            return cleaned_id
        else:
            return url_or_id  # Return as-is and let the API handle the error

def get_youtube_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript, None
    except TranscriptsDisabled:
        return None, "No captions/transcripts available for this video. This could be because:\n- The video is a live stream\n- Captions are disabled\n- The video is private/restricted"
    except Exception as e:
        error_msg = str(e)
        if "invalid video id" in error_msg.lower():
            return None, f"Invalid video ID '{video_id}'. Please check the URL/ID and try again."
        elif "could not retrieve a transcript" in error_msg.lower():
            return None, "Could not retrieve transcript. This might be a live stream, private video, or captions may be disabled."
        else:
            return None, f"Error getting transcript: {error_msg}"

def format_docs(retrieved_docs):
    """Format retrieved documents for context"""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Initialize session state for vector store
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
    st.session_state.current_video_id = None

st.title("YouTube Video Q&A with RAG")
st.write("Enter a YouTube video URL or ID to chat with the video content!")

# Video input section
video_input = st.text_input("Enter YouTube Video URL or ID:", placeholder="https://youtube.com/watch?v=dQw4w9WgXcQ or dQw4w9WgXcQ")


# Storage option selection
storage_option = st.selectbox(
    "Choose vector storage:",
    ["FAISS (Local)", "Pinecone (Cloud)"]
)

if st.button("Process Video") and video_input:
    video_id = extract_video_id(video_input.strip())
    
    with st.spinner("Extracting transcript..."):
        transcript, error = get_youtube_transcript(video_id)
        
        if error:
            st.error(error)
        else:
            st.success("Transcript extracted successfully!")
            
            # Process transcript
            with st.spinner("Processing and creating embeddings..."):
                # Split transcript into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])
                
                # Create embeddings
                embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                
                # Create vector store based on selection
                if storage_option == "FAISS (Local)":
                    st.session_state.vector_store = FAISS.from_documents(chunks, embedding)
                    st.session_state.retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 4}
                    )
                else:  # Pinecone
                    if pinecone_api:
                        pc = Pinecone(pinecone_api)
                        index = pc.Index("youtubechatbot")
                        st.session_state.vector_store = PineconeVectorStore(
                            embedding=embedding,
                            index=index
                        )
                        st.session_state.retriever = st.session_state.vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 4}
                        )
                    else:
                        st.error("Pinecone API key not found! Please set PINECONE_API_KEY in your environment.")
                        st.stop()
                
                st.session_state.vector_store_ready = True
                st.session_state.current_video_id = video_id
                
            st.success("Vector store created successfully! You can now ask questions about the video.")

# Q&A Section
if st.session_state.vector_store_ready:
    st.subheader(f"Chat with Video: {st.session_state.current_video_id}")
    
    # Initialize LLM
    if groq_api_key:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    else:
        st.error("Groq API key not found! Please set GROQ_API_KEY in your environment.")
        st.stop()
    
    # Create prompt template - use 'input' instead of 'question' to match retrieval chain
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant. Answer ONLY from the provided transcript context. 
        If the context is insufficient, just say you don't know.

        Context: {context}
        
        Question: {input}
        
        Answer:
        """,
        input_variables=['context', 'input']
    )
    
    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
    
    # Question input
    question = st.text_input("Ask a question about the video:", placeholder="What is the main topic of this video?")
    
    if question:
        with st.spinner("Generating answer..."):
            start_time = time.process_time()
            
            try:
                response = retrieval_chain.invoke({"input": question})
                response_time = time.process_time() - start_time
                
                st.write("**Answer:**")
                st.write(response['answer'])
                
                st.info(f"Response time: {response_time:.2f} seconds")
                
                # Show relevant chunks in expander
                with st.expander("📄 Source Context from Video"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")
                        
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

else:
    st.info("👆 Please process a video first to start asking questions!")

# Sidebar with information
