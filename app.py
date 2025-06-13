import streamlit as st
from streamlit_chat import message # For chat bubbles
from services.document_service import process_documents
from services.rag_service import create_vector_store, get_rag_response, PROVIDED_API_KEY as DEFAULT_RAG_API_KEY

# --- Page Configuration ---
st.set_page_config(
    page_title="DocuChat RAG",
    page_icon="\U0001F4AC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Inter & Poppins inspired) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600&display=swap');
    
    html, body, [class*="st-"], .main .block-container {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        border-radius: 0.5rem;
        padding: 0.5em 1em;
        font-family: 'Poppins', sans-serif;
    }
    .stFileUploader label {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    /* Chat bubble styling */
    .stChatMessage {
        border-radius: 0.75rem;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "uploaded_files_cache" not in st.session_state: # To track if files changed
    st.session_state.uploaded_files_cache = [] 
if "processed_chunks_with_metadata" not in st.session_state:
    st.session_state.processed_chunks_with_metadata = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "all_chunk_texts" not in st.session_state: # Store only the texts for RAG context
    st.session_state.all_chunk_texts = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = [] # {role: 'user'/'assistant', content: '...'} 

# --- Sidebar for API Key and Controls ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg", width=200) # Placeholder logo
    st.markdown("## DocuChat RAG \U0001F4AC")
    st.markdown("Upload your PDFs and chat with them using AI!")

    st.markdown("### Configuration")
    # Use the API key from rag_service as default, but allow user override
    user_api_key = st.text_input("Enter your Google API Key:", type="password", value=st.session_state.api_key)
    if user_api_key:
        st.session_state.api_key = user_api_key

    st.markdown("---    ")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="You can upload multiple PDF files here."
    )

    if st.button("Process Documents", use_container_width=True, disabled=not uploaded_files or not st.session_state.api_key):
        if not st.session_state.api_key:
            st.error("Please enter your Google API Key to proceed.")
        elif uploaded_files:
            # Check if files are new or changed to avoid reprocessing
            current_file_ids = sorted([f.file_id for f in uploaded_files])
            if current_file_ids != st.session_state.uploaded_files_cache:
                st.session_state.messages = [] # Reset chat if new docs are processed
                st.session_state.processing_complete = False
                with st.spinner("Processing documents... This may take a moment."):
                    try:
                        st.session_state.processed_chunks_with_metadata = process_documents(uploaded_files)
                        if not st.session_state.processed_chunks_with_metadata:
                            st.error("No text could be extracted from the documents, or the documents are empty.")
                        else:
                            st.session_state.all_chunk_texts = [item['chunk_text'] for item in st.session_state.processed_chunks_with_metadata]
                            st.session_state.vector_store, _ = create_vector_store(st.session_state.processed_chunks_with_metadata, st.session_state.api_key)
                            if st.session_state.vector_store:
                                st.session_state.processing_complete = True
                                st.session_state.uploaded_files_cache = current_file_ids
                                st.success("Documents processed successfully! You can now chat with them.")
                            else:
                                st.error("Failed to create vector store. Check logs or API key.")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        st.session_state.processing_complete = False
            else:
                st.info("These documents have already been processed. You can start chatting or upload new ones.")
                st.session_state.processing_complete = True # Ensure chat is enabled
        else:
            st.warning("Please upload at least one PDF file.")

    if st.session_state.processing_complete and st.session_state.messages:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# --- Main Chat Area ---
st.header("Chat with your Documents")

if not st.session_state.api_key:
    st.warning("Please enter your Google API Key in the sidebar to begin.")

if not uploaded_files and not st.session_state.processing_complete:
    st.info("Upload PDF files and click 'Process Documents' in the sidebar to start chatting.")

# Display chat messages
for i, chat_message in enumerate(st.session_state.messages):
    is_user = chat_message["role"] == "user"
    # Using streamlit-chat's message component
    message(chat_message["content"], is_user=is_user, key=f"chat_msg_{i}", avatar_style="thumbs" if not is_user else "initials", seed="AI" if not is_user else "U")

# Chat input
if st.session_state.processing_complete:
    user_query = st.chat_input("Ask a question about your documents...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        message(user_query, is_user=True, key=f"chat_msg_{len(st.session_state.messages)}_user", avatar_style="initials", seed="U")
        
        with st.spinner("Thinking..."):
            try:
                response = get_rag_response(user_query, st.session_state.vector_store, st.session_state.all_chunk_texts, st.session_state.api_key)
                st.session_state.messages.append({"role": "assistant", "content": response})
                message(response, is_user=False, key=f"chat_msg_{len(st.session_state.messages)}_assistant", avatar_style="thumbs", seed="AI")
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                message(error_msg, is_user=False, key=f"chat_msg_{len(st.session_state.messages)}_error", avatar_style="thumbs", seed="AI")
                st.error(error_msg)
else:
    if uploaded_files and st.session_state.api_key:
        st.info("Click 'Process Documents' in the sidebar to enable chat.")
    elif not uploaded_files and st.session_state.api_key:
        st.info("Please upload your PDF documents to get started.")
