import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Set your Groq API key (for demo, you may want to use st.secrets or .env in production)
os.environ["GROQ_API_KEY"] = ""

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")
st.title("YouTube Video RAG Chatbot")
st.write("""
This app lets you ask questions about a YouTube video's transcript using Retrieval-Augmented Generation (RAG) with a local embedding model and Groq's Llama3-8b-8192 LLM.
""")

with st.sidebar:
    st.header("YouTube Video")
    video_id = st.text_input("Enter YouTube Video ID", value="Gfr50f6ZBvo")
    fetch_button = st.button("Fetch Transcript & Build Index")

@st.cache_data(show_spinner=True)
def fetch_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        # Try notebook-style: fetched.snippets
        if hasattr(fetched, 'snippets'):
            transcript_text = " ".join(snippet.text for snippet in fetched.snippets)
        # Fallback: fetched is likely a list of dicts
        elif isinstance(fetched, list) and 'text' in fetched[0]:
            transcript_text = " ".join(entry["text"] for entry in fetched)
        else:
            # fallback to get_transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript_text = " ".join([entry["text"] for entry in transcript])
        return transcript_text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

@st.cache_resource(show_spinner=True)
def build_vector_store(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=transcript_text, metadata={"source": "youtube"})]
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, chunks

def format_docs(retrieved_docs):
    if isinstance(retrieved_docs, list):
        context_text = " ".join(doc.page_content for doc in retrieved_docs)
    elif isinstance(retrieved_docs, Document):
        context_text = retrieved_docs.page_content
    else:
        context_text = str(retrieved_docs)
    return context_text

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer only from the provided transcript context.
    If the context is insufficient, just say you don't know.
    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

llm = ChatGroq(model_name="llama3-8b-8192")
parser = StrOutputParser()

if fetch_button and video_id:
    with st.spinner("Fetching transcript and building vector store..."):
        transcript_text = fetch_transcript(video_id)
        if transcript_text:
            vector_store, chunks = build_vector_store(transcript_text)
            st.session_state["vector_store"] = vector_store
            st.session_state["chunks"] = chunks
            st.success(f"Transcript loaded and indexed! {len(chunks)} chunks created.")
        else:
            st.session_state["vector_store"] = None
            st.session_state["chunks"] = None

if "vector_store" in st.session_state and st.session_state["vector_store"]:
    st.subheader("Ask a question about the video")
    user_question = st.text_input("Your question:", value="What is DeepMind?")
    ask_button = st.button("Get Answer")

    if ask_button and user_question:
        with st.spinner("Retrieving and generating answer..."):
            retriever = st.session_state["vector_store"].as_retriever(search_type="similarity", search_kwargs={"k": 5})
            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            main_chain = parallel_chain | prompt | llm | parser
            try:
                answer = main_chain.invoke(user_question)
                st.markdown(f"**Answer:**\n{answer}")
            except Exception as e:
                st.error(f"Error during answer generation: {e}")

    with st.expander("Show transcript chunks"):
        for i, chunk in enumerate(st.session_state["chunks"]):
            st.markdown(f"**Chunk {i+1}:** {chunk.page_content[:300]}{'...' if len(chunk.page_content) > 300 else ''}")
else:
    st.info("Enter a YouTube video ID and click 'Fetch Transcript & Build Index' to get started.")
