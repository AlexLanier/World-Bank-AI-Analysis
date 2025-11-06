"""
RAG (Retrieval-Augmented Generation) engine for document retrieval
Uses ChromaDB for vector storage and retrieval
"""
import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import hashlib

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Sentence transformers not available: {e}. Using ChromaDB default embeddings.")
    EMBEDDINGS_AVAILABLE = False
    embedding_model = None

# Initialize ChromaDB
CHROMA_PERSIST_DIR = "./chroma_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

client = None
collection = None

try:
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name="worldbank_docs",
        metadata={"description": "World Bank loan data and documentation"}
    )
    print("✅ ChromaDB initialized successfully")
except Exception as e:
    print(f"⚠️  Error initializing ChromaDB: {e}")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def index_file(file_path: str, file_type: str = "text") -> Dict:
    """
    Index a file into ChromaDB
    
    Args:
        file_path: Path to the file
        file_type: Type of file ('text', 'csv', 'markdown')
        
    Returns:
        Dict with 'success', 'chunks_indexed', and 'message'
    """
    if not collection:
        return {
            'success': False,
            'chunks_indexed': 0,
            'message': 'ChromaDB not initialized'
        }
    
    try:
        # Generate file ID
        file_id = hashlib.md5(file_path.encode()).hexdigest()
        
        # Check if already indexed
        existing = collection.get(ids=[file_id])
        if existing['ids']:
            return {
                'success': True,
                'chunks_indexed': 0,
                'message': f'File already indexed: {file_path}'
            }
        
        chunks = []
        chunk_ids = []
        metadata_list = []
        
        if file_type == 'csv':
            # Read CSV and convert to text
            df = pd.read_csv(file_path)
            text = f"CSV file: {os.path.basename(file_path)}\n\n"
            text += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            text += f"Shape: {df.shape}\n\n"
            text += "Sample data:\n"
            text += df.head(10).to_string()
            
            chunks = chunk_text(text)
            
        elif file_type == 'markdown':
            # Read markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunk_text(text)
            
        else:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunk_text(text)
        
        # Create IDs and metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_{i}"
            chunk_ids.append(chunk_id)
            metadata_list.append({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_type': file_type,
                'chunk_index': i
            })
        
        # Add to collection (ChromaDB will handle embeddings automatically)
        try:
            collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=metadata_list
            )
        except Exception as e:
            # If embedding fails, try without custom embeddings
            print(f"⚠️  Error adding to collection: {e}")
            return {
                'success': False,
                'chunks_indexed': 0,
                'message': f'Error adding chunks: {str(e)}'
            }
        
        return {
            'success': True,
            'chunks_indexed': len(chunks),
            'message': f'Successfully indexed {len(chunks)} chunks from {file_path}'
        }
        
    except Exception as e:
        return {
            'success': False,
            'chunks_indexed': 0,
            'message': f'Error indexing file: {str(e)}'
        }


def index_folder(folder_path: str = "./rag_files") -> Dict:
    """
    Index all files in a folder
    
    Args:
        folder_path: Path to folder containing files to index
        
    Returns:
        Dict with 'success', 'files_indexed', and 'details'
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return {
            'success': True,
            'files_indexed': 0,
            'details': f'Created folder: {folder_path}'
        }
    
    results = {
        'success': True,
        'files_indexed': 0,
        'details': []
    }
    
    # Index all files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            file_type = 'text'
            if file_ext == '.csv':
                file_type = 'csv'
            elif file_ext in ['.md', '.markdown']:
                file_type = 'markdown'
            elif file_ext in ['.txt', '.text']:
                file_type = 'text'
            else:
                continue  # Skip unsupported file types
            
            result = index_file(file_path, file_type)
            results['details'].append({
                'file': file_path,
                'result': result
            })
            if result['success']:
                results['files_indexed'] += 1
    
    return results


def query_rag(query: str, n_results: int = 3) -> Dict:
    """
    Query the RAG system for relevant documents
    
    Args:
        query: User's query
        n_results: Number of results to return
        
    Returns:
        Dict with 'success', 'results', and 'context'
    """
    if not collection:
        return {
            'success': False,
            'results': [],
            'context': None,
            'message': 'ChromaDB not initialized'
        }
    
    try:
        # Query collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        if results['documents'] and len(results['documents'][0]) > 0:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            # Build context string
            context_parts = []
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                file_name = metadata.get('file_name', 'Unknown')
                context_parts.append(f"[Source: {file_name}]\n{doc}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            return {
                'success': True,
                'results': documents,
                'metadatas': metadatas,
                'context': context,
                'message': f'Found {len(documents)} relevant documents'
            }
        else:
            return {
                'success': True,
                'results': [],
                'context': None,
                'message': 'No relevant documents found'
            }
            
    except Exception as e:
        return {
            'success': False,
            'results': [],
            'context': None,
            'message': f'Error querying RAG: {str(e)}'
        }


# Initialize on import
if collection:
    index_folder("./rag_files")

