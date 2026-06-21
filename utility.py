import chromadb
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import sys
from sentence_transformers import CrossEncoder
from pathlib import Path
import ollama


def progress_bar(current, total, bar_length=40):
    fraction = current / total
    arrow = int(fraction * bar_length) * '█'
    padding = (bar_length - len(arrow)) * '-'
    ending = '\n' if current == total else ''
    
    # \r moves the cursor back to the start of the line
    sys.stdout.write(f'\rProgress: |{arrow}{padding}| {int(fraction * 100)}%')
    sys.stdout.flush()
    if current == total:
        print()

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5"
)


client=chromadb.PersistentClient(path="./chroma_db")
collection=client.get_or_create_collection(name="default", embedding_function=embedding_function)
model= CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    
def pdf_to_database(pdf_path="default.pdf"):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=500
    )
    pdf_name = Path(pdf_path).name

    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception as e:
        print(f"Failed to read PDF: {e}")
        return
    
    full_text = ""
    page_bounds = []
    total_pages = len(reader.pages)
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        progress_bar(page_num, total_pages)
        
        start_idx = len(full_text)
        full_text += text
        end_idx = len(full_text)
        
        page_bounds.append({
            "page": page_num,
            "start": start_idx,
            "end": end_idx
        })
        
    if not full_text.strip():
        print("\nNo text could be extracted from the PDF.")
        return

   
    chunks = text_splitter.split_text(full_text)

    
    documents, metadatas, ids = [], [], []
    current_search_idx = 0
    BATCH_SIZE = 50  
    for chunk_id, chunk in enumerate(chunks):
        chunk_start = full_text.find(chunk, current_search_idx)
        if chunk_start == -1:
            chunk_start = full_text.find(chunk)
            
        if chunk_start != -1:
            current_search_idx = chunk_start + len(chunk)
        else:
            chunk_start = 0
            
        chunk_end = chunk_start + len(chunk)

        chunk_page = 1
        max_overlap = 0
        for bound in page_bounds:
            overlap_start = max(chunk_start, bound["start"])
            overlap_end = min(chunk_end, bound["end"])
            overlap_len = overlap_end - overlap_start
            
            if overlap_len > max_overlap:
                max_overlap = overlap_len
                chunk_page = bound["page"]

        
        documents.append(chunk)
        metadatas.append({
            "source": pdf_name,
            "page": chunk_page,  
            "created_at": time.time()
        })
        ids.append(f"{pdf_name}_chunk_{chunk_id}")

        if len(documents) >= BATCH_SIZE:
            collection.add(
                documents=documents, 
                metadatas=metadatas, 
                ids=ids
            )
            documents.clear()
            metadatas.clear()
            ids.clear()

    if documents:
        collection.add(
            documents=documents, 
            metadatas=metadatas, 
            ids=ids
        )
        documents.clear()
        metadatas.clear()
        ids.clear()
        
    print(f"\nSuccessfully indexed {len(chunks)} chunks from {pdf_name} with pixel-perfect page matching.")

def delete_from_database(pdf_path):
    
    pdf_name=Path(pdf_path).name
    collection.delete(where={"source":pdf_name})
    print(f"Deleted all entries from {pdf_name} in the database")

def query_to_database(query):
    
    result= collection.query(
        query_texts=[query],
        n_results=12
    
    )
    return result

def specific_query_to_database(query, pdf_name):
    result= collection.query(
        query_texts=[query],
        n_results=12,
        where={"source":pdf_name}
    )
    return result

def reranker(query,res,top_k=5):

    x = res['documents'][0]
    
    scores = model.predict([(query, doc) for doc in x])
    sorted_indices_with_score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    top_k_result = [index for index, score in sorted_indices_with_score]
    
    sorted_res = {}
    for key in res.keys():
        if isinstance(res[key], list) and len(res[key]) > 0 and isinstance(res[key][0], list) and len(res[key][0]) > 0:
            sorted_res[key] = [[res[key][0][i] for i in top_k_result if i < len(res[key][0])]]
        else:
            # Safely keeps metadata fields, None values, or empty fields intact
            sorted_res[key] = res[key]
            
    return sorted_res

def get_response():
    query=input("Enter your query: ")

    choice=input("Do you want to see the results from the specific PDF? (yes/no): ")
    if choice.lower() == "yes":
        source=input("Enter the source PDF name (e.g., Beiser_Modern_Physics (1).pdf): ")
        res=specific_query_to_database(query, source)
    else:    res=query_to_database(query)

    result=reranker(query,res)


    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    context = ""

    for doc, meta in zip(documents, metadatas):
        context += (
            f"[Source: {meta['source']}, Page {meta['page']}]\n"
            f"{doc}\n\n"
        )


    sources = {}

    for meta in metadatas:
        file = meta["source"]
        page = meta["page"]

        if file not in sources:
            sources[file] = set()

        sources[file].add(page)


    source_text = []

    for file, pages in sources.items():
        pages = sorted(pages)

        if len(pages) == 1:
            page_string = f"Page {pages[0]}"
        else:
            page_string = "Pages " + ", ".join(str(p) for p in pages)

        source_text.append(f"{file} ({page_string})")


    source_line = "Sources: " + " | ".join(source_text)

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Act as a professional writer. "
                    "I need you to write a detailed, larger paragraph. Immediately following "
                    "the paragraph, extract the key points and present them separately as a clear, bulleted list. "
                    "Answer the question only using the provided context. "
                    "If the information is not present in the context, say so clearly."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n"
                    f"Question: {query}"
                )
            }
        ],
        stream=True  
    )

    print(source_line)

    for chunk in response:
        # Print each word/token as it arrives without adding a new line
        print(chunk['message']['content'], end='', flush=True)


    print()

