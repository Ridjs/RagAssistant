from pypdf import PdfReader

class DocumentParser:
    @staticmethod
    def extract_text(file) -> str:
        read = PdfReader(file)
        text = ""
        for page_num, page in enumerate(read.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n" + page_text
        return text

    @staticmethod
    def create_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:

        if chunk_size <= chunk_overlap:
            raise ValueError("Chunk size must be strictly greater than chunk overlap")

        chunks= []
        start=0
        step = chunk_size - chunk_overlap

        while start < len(text):
            end = start + chunk_overlap
            chunk = text[start:end]
            cleaned_chunk = chunk.strip();
            if cleaned_chunk:
                chunks.append(cleaned_chunk)
            start+=step

        return chunks