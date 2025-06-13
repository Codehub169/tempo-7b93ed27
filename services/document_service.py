import fitz  # PyMuPDF

def parse_pdf_files(uploaded_files):
    """
    Parses uploaded PDF files and extracts text content from each.

    Args:
        uploaded_files: A list of Streamlit UploadedFile objects.

    Returns:
        A list of dictionaries, where each dictionary contains:
        'file_name': The name of the PDF file.
        'text_content': The extracted text content from the PDF.
    """
    documents = []
    for uploaded_file in uploaded_files:
        try:
            # To read file-like objects, PyMuPDF needs bytes
            file_bytes = uploaded_file.getvalue()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_content = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            documents.append({"file_name": uploaded_file.name, "text_content": text_content})
            doc.close()
        except Exception as e:
            print(f"Error parsing {uploaded_file.name}: {e}")
            documents.append({"file_name": uploaded_file.name, "text_content": "", "error": str(e)})
    return documents

def chunk_text(text_content, chunk_size=1000, chunk_overlap=200):
    """
    Splits a given text content into smaller, overlapping chunks.

    Args:
        text_content (str): The text to be chunked.
        chunk_size (int): The desired size of each chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        A list of text chunks (strings).
    """
    if not text_content:
        return []
    
    chunks = []
    start = 0
    while start < len(text_content):
        end = start + chunk_size
        chunks.append(text_content[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text_content) and end < len(text_content): # Ensure last part is captured if loop condition breaks early
            chunks.append(text_content[start:]) 
    # A small correction if the last chunk added by the loop logic is an empty string from overlap exceeding remaining text
    if chunks and not chunks[-1].strip():
        chunks.pop()
    # Ensure the very last piece of text is captured if overlap logic makes it skip
    if len(text_content) > (start - (chunk_size - chunk_overlap)) and start < len(text_content):
        last_chunk_start = start - (chunk_size - chunk_overlap) # Re-evaluate potential start of last segment
        if last_chunk_start + chunk_size > len(text_content) and last_chunk_start < len(text_content):
             # Check if the last chunk is already effectively the end of the text
            if not chunks or chunks[-1] != text_content[last_chunk_start:]:
                 if text_content[last_chunk_start:].strip():
                    chunks.append(text_content[last_chunk_start:])
    
    # Remove duplicates that might arise from overlap logic at the end
    if len(chunks) > 1 and chunks[-1] == chunks[-2]:
        chunks.pop()
        
    return [chunk for chunk in chunks if chunk.strip()] # Filter out any purely whitespace chunks

def process_documents(uploaded_files, chunk_size=1000, chunk_overlap=200):
    """
    Parses uploaded PDF files, extracts text, and splits the text into chunks.

    Args:
        uploaded_files: A list of Streamlit UploadedFile objects.
        chunk_size (int): The desired size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        A list of text chunks, with each chunk associated with its source file name.
        Each element is a dict: {'file_name': str, 'chunk_text': str}
    """
    parsed_docs = parse_pdf_files(uploaded_files)
    all_chunks_with_metadata = []
    for doc_data in parsed_docs:
        if "error" in doc_data:
            print(f"Skipping chunking for {doc_data['file_name']} due to parsing error.")
            continue
        
        text_chunks = chunk_text(doc_data['text_content'], chunk_size, chunk_overlap)
        for chunk in text_chunks:
            all_chunks_with_metadata.append({"file_name": doc_data['file_name'], "chunk_text": chunk})
    return all_chunks_with_metadata
