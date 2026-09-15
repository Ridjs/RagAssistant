import re
import pymupdf

class DocumentParser:
    @staticmethod
    def extract_text(file) -> str:
        read = pymupdf.open(stream=file,filetype="pdf")
        text = ""
        for page_num, page in enumerate(read):
            page_text = page.get_text("text",sort=True)
            if page_text:
                cleaned_text = re.sub(r'\s+',' ', page_text).strip()
                if cleaned_text:
                    text += f"\n--- Page {page_num + 1} ---\n" + cleaned_text            
        read.close()
        return text

    @staticmethod
    def create_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:

        if chunk_size <= chunk_overlap:
            raise ValueError("Chunk size must be strictly greater than chunk overlap")

        chunks= []
        start=0
        step = chunk_size - chunk_overlap

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            cleaned_chunk = chunk.strip();
            if len(cleaned_chunk) > 10:
                chunks.append(cleaned_chunk)
            start+=step

        return chunks