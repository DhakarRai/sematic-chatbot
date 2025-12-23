def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into chunks.
    For small data: splits by lines (each line = 1 chunk)
    For large data: splits by character count with overlap
    """
    # First, try splitting by lines (better for small data)
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    # If we have multiple meaningful lines, use line-based chunking
    if len(lines) >= 3:
        return lines
    
    # For large text without clear line breaks, use character-based chunking
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks
