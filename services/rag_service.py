import google.generativeai as genai
import numpy as np
import faiss

# Store the API key provided by the user
# In a production app, use Streamlit secrets or environment variables
PROVIDED_API_KEY = ""

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

        # 1. Check for explicit blocking at the prompt level
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason_message = f"Response blocked due to: {response.prompt_feedback.block_reason.name}"
            if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings:
                ratings_info = [
                    f"{rating.category.name.lower()} ({rating.probability.name.lower()})"
                    for rating in response.prompt_feedback.safety_ratings
                ]
                if ratings_info:
                    block_reason_message += f" (Details: {'; '.join(ratings_info)})"
            return block_reason_message

        # 2. If not blocked at prompt level, check the candidates
        # response.text will attempt to access candidates[0].content.parts[0].text
        # It will raise an error if candidates is empty or content is not text.
        if not response.candidates:
            return "Sorry, the model did not generate a response. This might be due to safety filters or other issues. Please try rephrasing."

        first_candidate = response.candidates[0]
        
        # Check finish reason of the first candidate
        # Valid finish reasons include STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
        if first_candidate.finish_reason.name != "STOP":
            safety_details = ""
            if first_candidate.finish_reason.name == "SAFETY" and hasattr(first_candidate, 'safety_ratings') and first_candidate.safety_ratings:
                ratings_info = [
                    f"{rating.category.name.lower()} ({rating.probability.name.lower()})"
                    for rating in first_candidate.safety_ratings if rating.probability.value > 2 # Often 0: unspecified, 1: negligible, 2: low, 3: medium, 4: high
                ]
                if ratings_info:
                    safety_details = f" (Safety issues: {'; '.join(ratings_info)})"
            return f"Sorry, the response could not be fully generated. Reason: {first_candidate.finish_reason.name}{safety_details}. Try rephrasing your query or checking the document content."

        # 3. If finish_reason is STOP, try to get the text.
        # The .text property itself might raise an error if the content part is missing or not text.
        try:
            if response.text: # response.text can raise ValueError if content isn't text-based
                return response.text
            # If response.text is empty but no error, check parts manually (more robust for complex content)
            elif first_candidate.content and first_candidate.content.parts:
                all_parts_text = "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text'))
                if all_parts_text:
                    return all_parts_text
                return "The model generated a response, but it contained no readable text. Please try again."
            else: # Candidate exists, finish reason STOP, but no text and no parts.
                return "The model generated an empty response. Please try rephrasing your query."
        except ValueError as ve: # Specifically catch ValueError from response.text if content is not text
            print(f"ValueError accessing response.text: {ve}. Candidate content: {first_candidate.content}")
            return "Sorry, the model's response was not in a readable text format. Please try again."


    except Exception as e:
        # This catches errors from model.generate_content() itself (e.g., API key, network)
        # or any other unexpected errors during response processing.
        print(f"Error in generate_response: {type(e).__name__} - {e}")
        # Provide a more specific error to the user.
        return f"Sorry, I encountered an error ({type(e).__name__}): {str(e)}. Please check your API key, network connection, or try again later."


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
        # Check if vector store or all_chunk_texts might be empty leading to no relevant_chunks
        if not all_chunk_texts:
             return "No documents have been processed yet, or they contained no text. Please upload and process documents first."
        return "I couldn't find any relevant information in the uploaded documents to answer your question. Try asking something more general about the documents or rephrasing."
    
    return generate_response(query, relevant_chunks, api_key)
