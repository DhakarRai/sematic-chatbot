def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into chunks.

    - For small data:
      Splits by lines (each meaningful line = 1 chunk)

    - For large data:
      Splits by character count with overlap
    """

    if not text:
        return []

    # First, try splitting by lines (best for small structured data)
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    # If multiple meaningful lines exist, use line-based chunking
    if len(lines) >= 3:
        return lines

    # Otherwise, use character-based chunking (for long text)
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, 0)

    return chunks
