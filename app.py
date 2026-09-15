import os
import streamlit as st
from document_parser import DocumentParser
from vector_store import VectorStore
from groq import Groq

st.set_page_config(page_title="Personal RAG Engine",layout="wide")
st.title("Personal Document RAG Engine")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

vs = st.session_state.vector_store

st.sidebar.header("Indexing and Retrievel Settings")
chunk_size = st.sidebar.slider("Chunk Size (Characters)", 200, 1000, 500, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap (Characters)",0,200,50,step=10)
top_k = st.sidebar.slider("Top-K Retrievel Chunks",1,5,3)

uploaded_file = st.file_uploader("Upload a PDF document to begin:", type="pdf")
if uploaded_file:
    cache_filename = f"{uploaded_file.name}_cache.npz"
    col1,col2 = st.columns([1,1])
    with col1:
        if st.button("Load Cached Index from Disk"):
            if vs.load_cache(cache_filename):
                st.success(f"Loaded binary vector index from '{cache_filename}'!")
            else:
                st.error("No local .npz binary cache found for this document.")
    with col2:
        if st.button("Build New Index"):
            with st.spinner("Extracting text, chunking, and encoding vectors"):
                text = DocumentParser.extract_text(uploaded_file)
                chunks = DocumentParser.create_chunks(text,chunk_size,chunk_overlap)

                vs.index(chunks)
                vs.save_cache(cache_filename)
                st.success(f"Indexed {len(chunks)} chunks & saved binary cache to disk!")

    st.divider()
    user_query= st.text_input("Ask a question based on indexed content:", max_chars=500)

    if user_query and st.button("Generate Answer"):
        results = vs.search(user_query,top_k=top_k)
        if not results:
            st.warning("Index is empty. Please build or load a vector index first.")
        else:
            context_blocks = [chunk for chunk, score in results]
            combined_context = "\n\n".join(context_blocks)
            prompt = f"Context:\n{combined_context}\n\nQuestion: {user_query}\nAnswer accurately using ONLY the context provided above"
            
            with st.spinner("Generating answer..."):
                try:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    response = client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role":"user","content":prompt}],max_tokens=300,reasoning_effort="low").choices[0].message.content
                    st.subheader("Answer:")
                    st.write(response)
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        st.warning("This app is getting a lot of use right now - try again in a minute.")
                    else:
                        st.error("Something went wrong generating the answer.")

            with st.expander("Inspect Top-K Retrieved Context"):
                for idx, (chunk, score) in enumerate(results):
                    st.markdown(f"**Chunk {idx+1} (Similarity Score: {score:.4f})**")
                    st.code(chunk, language="text")