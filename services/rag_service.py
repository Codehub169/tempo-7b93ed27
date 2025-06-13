import google.generativeai as genai
import numpy as np
import faiss

# Store the API key provided by the user
# In a production app, use Streamlit secrets or environment variables
PROVIDED_API_KEY = "AIzaSyCAH4QbvHnjo4hQXKMAhaI9KP8gr3WVMB4"

MODEL_NAME = "gemini-1.5-flash-preview-0514" # As per plan, using a close available model
EMBEDDING_MODEL_NAME = "models/embedding-001"

def configure_gemini_client(api_key):
    """Configures the Google Generative AI client with the provided API key."""
    if not api_key:
        raise ValueError("API key for Google Generative AI is not provided.")
    genai.configure(api_key=api_key)

def embed_texts(texts, api_key):
    """
    Generates embeddings for a list of text chunks using Gemini API.

    Args:
        texts (list): A list of text strings to embed.
        api_key (str): The Google API key.

    Returns:
        A list of embeddings (numpy arrays), or None if an error occurs.
    """
    configure_gemini_client(api_key)
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=texts, task_type="RETRIEVAL_DOCUMENT")
        return [np.array(embedding) for embedding in result['embedding']]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def create_vector_store(text_chunks_with_metadata, api_key):
    """
    Creates a FAISS vector store from text chunks.

    Args:
        text_chunks_with_metadata (list): List of dicts, each {'file_name': str, 'chunk_text': str}.
        api_key (str): The Google API key.

    Returns:
        A FAISS index and the list of chunk texts, or (None, None) if an error occurs.
    """
    if not text_chunks_with_metadata:
        return None, []

    chunk_texts = [item['chunk_text'] for item in text_chunks_with_metadata]
    embeddings = embed_texts(chunk_texts, api_key)
    if embeddings is None or not embeddings:
        print("Failed to generate embeddings for vector store creation.")
        return None, chunk_texts

    embeddings_np = np.array(embeddings).astype('float32')
    if embeddings_np.ndim == 1: # Handle case of single chunk
        embeddings_np = np.expand_dims(embeddings_np, axis=0)
    
    if embeddings_np.shape[0] == 0: # No embeddings were successfully created
        print("No embeddings available to create vector store.")
        return None, chunk_texts

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return index, chunk_texts # Return both index and the original texts for context retrieval

def search_vector_store(query, vector_store, all_chunk_texts, api_key, k=5):
    """
    Searches the vector store for chunks similar to the query.

    Args:
        query (str): The user's query.
        vector_store (faiss.Index): The FAISS index.
        all_chunk_texts (list): List of all original text chunks.
        api_key (str): The Google API key.
        k (int): The number of top similar chunks to retrieve.

    Returns:
        A list of relevant text chunks.
    """
    if not query or vector_store is None:
        return []

    query_embedding_list = embed_texts([query], api_key) # Embed as a list
    if not query_embedding_list:
        print("Failed to embed query.")
        return []
    
    query_embedding = np.array(query_embedding_list[0]).astype('float32').reshape(1, -1)
    
    # Ensure k is not greater than the number of items in the index
    num_indexed_items = vector_store.ntotal
    actual_k = min(k, num_indexed_items)
    if actual_k == 0:
        return []

    distances, indices = vector_store.search(query_embedding, actual_k)
    
    relevant_chunks = [all_chunk_texts[i] for i in indices[0] if i < len(all_chunk_texts)]
    return relevant_chunks

def generate_response(query, context_chunks, api_key):
    """
    Generates a response using Gemini based on the query and context.

    Args:
        query (str): The user's query.
        context_chunks (list): A list of text chunks to use as context.
        api_key (str): The Google API key.

    Returns:
        The AI-generated response string, or an error message string.
    """
    configure_gemini_client(api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = (
        f"You are a helpful assistant. Answer the following question based ONLY on the provided context.\n"
        f"If the answer cannot be found in the context, clearly state that the information is not in the documents.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response from Gemini: {e}")
        # Check for specific safety/blockage reasons if the API provides them
        try:
            if response.prompt_feedback.block_reason:
                 return f"Response blocked due to: {response.prompt_feedback.block_reason}"
        except Exception:
            pass # If feedback is not available or error is different
        return "Sorry, I encountered an error while generating the response."

def get_rag_response(query, vector_store, all_chunk_texts, api_key):
    """
    Performs RAG to get a response for the query.

    Args:
        query (str): The user's query.
        vector_store (faiss.Index): The FAISS index.
        all_chunk_texts (list): List of all original text chunks.
        api_key (str): The Google API key.

    Returns:
        The AI-generated response.
    """
    if not api_key:
        return "API Key is not configured. Please set it up first."
    
    relevant_chunks = search_vector_store(query, vector_store, all_chunk_texts, api_key, k=5)
    if not relevant_chunks:
        return "I couldn't find any relevant information in the uploaded documents to answer your question."
    
    return generate_response(query, relevant_chunks, api_key)
