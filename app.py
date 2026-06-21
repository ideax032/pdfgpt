import streamlit as st
import os
from pathlib import Path
import pandas as pd

# Import backend variables and functions directly from your utility.py file
from utility_app import (
    pdf_to_database,
    delete_from_database,
    query_to_database,
    specific_query_to_database,
    reranker,
    prepare_context_and_citations,
    stream_llm_response,
    collection
)

# Page Setup
st.set_page_config(page_title="Local RAG Playground", layout="wide", page_icon="📚")
st.title("📚 Local RAG Learning Playground")
st.caption("A private self-use RAG application powered by ChromaDB, Cross-Encoders, and Ollama.")

# --- SIDEBAR: Settings & Document Processing ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Feature 1: Live System Prompt Tweaking for Learning
    st.subheader("💡 Prompt Engineering")
    system_prompt = st.text_area(
        "Modify LLM System Instructions:",
        value=(
            "You are a helpful assistant. Act as a professional writer. "
            "I need you to write a detailed, larger paragraph. Immediately following "
            "the paragraph, extract the key points and present them separately as a clear, bulleted list. "
            "Answer the question only using the provided context. "
            "If the information is not present in the context, say so clearly."
        ),
        height=200,
        help="Change this text to observe how system behaviors change your RAG outputs!"
    )
    
    st.write("---")
    
    # Document Upload Section
    st.subheader("📁 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF to Database", type=["pdf"])
    if uploaded_file is not None:
        temp_path = Path(uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        with st.spinner(f"Indexing '{uploaded_file.name}' with continuous page mapping..."):
            pdf_to_database(str(temp_path))
        st.success(f"Successfully indexed {uploaded_file.name}!")
        if temp_path.exists():
            os.remove(temp_path)
            st.rerun()
            
    st.write("---")
    
    # Feature 2: Database Diagnostics and Clear Controls
    st.subheader("📊 Database Diagnostics")
    try:
        db_data = collection.get(include=["metadatas", "documents"])
        total_chunks = len(db_data["ids"]) if db_data["ids"] else 0
        st.metric(label="Total Chunks in DB", value=total_chunks)
        
        if total_chunks > 0:
            sources = set([meta["source"] for meta in db_data["metadatas"]])
            
            file_to_delete = st.selectbox("Select file to delete:", list(sources))
            if st.button("🗑️ Delete Selected File", type="primary"):
                delete_from_database(file_to_delete)
                st.success(f"Deleted {file_to_delete} from DB.")
                st.rerun()
        else:
            st.info("ChromaDB is currently empty.")
    except Exception as e:
        st.error(f"Could not load database statistics: {e}")

# --- MAIN INTERFACE: Querying & Rerank Diagnostics ---
st.header("💬 Ask Your Documents")

# Extract existing indexed sources for target routing options
try:
    existing_meta = collection.get(include=["metadatas"])
    all_sources = list(set([m["source"] for m in existing_meta["metadatas"]])) if existing_meta["metadatas"] else []
except:
    all_sources = []

use_filter = st.checkbox("Focus query on a specific PDF file")
selected_source = None

if use_filter and all_sources:
    selected_source = st.selectbox("Select Target PDF:", all_sources)
elif use_filter and not all_sources:
    st.caption("⚠️ No documents indexed yet to focus on.")

query = st.text_input("Enter your query here:", placeholder="What does the document say about...")

if query:
    if total_chunks == 0:
        st.error("Please drop and index a PDF file in the sidebar panel before querying!")
    else:
        # Step 1: Query Retrieval Stage
        with st.spinner("Fetching matching vector spaces..."):
            if use_filter and selected_source:
                raw_res = specific_query_to_database(query, selected_source)
            else:
                raw_res = query_to_database(query)

        # Confirm we pulled vectors back
        if not raw_res or not raw_res.get('documents') or not raw_res['documents'][0]:
            st.warning("No matches found in database matching your query vectors.")
        else:
            # Step 2: Cross-Encoder Reranking
            with st.spinner("Evaluating contexts via Cross-Encoder Reranker..."):
                reranked_res = reranker(query, raw_res, top_k=5)
                
            # Step 3: Process text configurations using your utility transformations
            context_str, source_line = prepare_context_and_citations(reranked_res)

            # --- DIAGNOSTICS VISUALIZER PANEL ---
            with st.expander("🔍 Inspect RAG Pipeline Dynamics (Reranking Table)"):
                st.write("See how the initial Vector query matching ordered blocks vs how your Cross-Encoder sorted them:")
                
                raw_docs = raw_res['documents'][0]
                raw_metas = raw_res['metadatas'][0]
                final_docs = reranked_res['documents'][0]
                
                comparison_data = []
                for idx, (doc, meta) in enumerate(zip(raw_docs, raw_metas)):
                    # Check where this chunk ranked after cross-evaluation sorting
                    final_rank = "Dropped (Not in Top 5)"
                    for f_idx, f_doc in enumerate(final_docs):
                        if f_doc == doc:
                            final_rank = f"Passed to LLM (Rank #{f_idx + 1})"
                            break
                            
                    comparison_data.append({
                        "Initial Vector Rank": idx + 1,
                        "Original Page": meta['page'],
                        "Pipeline Outcome": final_rank,
                        "Text Chunk Snippet": doc[:140] + "..."
                    })
                    
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)

            # Render structured text source line on screen 
            st.write(f"**Verified Citations Found:**")
            st.caption(f"📖 {source_line}")
            st.write("---")
            
            # Step 4: Stream response directly to page interface
            st.write("**Generated Model Response:**")
            
            try:
                # Use Streamlit's native streaming engine with your backend generator function
                st.write_stream(stream_llm_response(query, context_str, system_prompt))
            except Exception as e:
                st.error(
                    f"Ollama execution error: {e}. "
                    "Make sure your local Ollama background engine is running with 'ollama run llama3'."
                )

# Optional bottom-drawer global database inspection grid
if st.sidebar.checkbox("👀 View Global Database Spreadsheet"):
    st.markdown("### Raw Database Entries")
    try:
        global_inspect = collection.get(include=["documents", "metadatas"])
        if global_inspect["ids"]:
            inspect_df = pd.DataFrame({
                "ID": global_inspect["ids"],
                "Source Document": [m["source"] for m in global_inspect["metadatas"]],
                "Mapped Page": [m["page"] for m in global_inspect["metadatas"]],
                "Raw Text Content": global_inspect["documents"]
            })
            st.dataframe(inspect_df, use_container_width=True)
        else:
            st.info("No data entries indexed to view.")
    except Exception as e:
        st.error(f"Could not build inspection table: {e}")