import os
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.chunks = []

    def index(self, chunks: list[str]):
        self.chunks = chunks
        raw_embeddings = self.model.encode(chunks, show_progress_bar=False)
        norms = np.linalg.norm(raw_embeddings,axis=1,keepdims=True)
        self.embeddings = raw_embeddings / np.maximum(norms, 1e-12)

    def search(self, query:str, top_k:int=3) -> list[tuple[str,float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_vec = self.model.encode([query])[0]
        query_norm = query_vec / max(np.linalg.norm(query_vec),1e-12)
        scores = np.dot(self.embeddings,query_norm)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def save_cache(self, file_path:str="vector_cache.npz"):
        if self.embeddings is not None:
            np.savez_compressed(
                file_path,
                embeddings = self.embeddings,
                chunks=np.array(self.chunks,dtype=object)
            )

    def load_cache(self, file_path:str="vector_cache.npz") -> bool:
        if os.path.exists(file_path):
            data = np.load(file_path,allow_pickle=True)
            self.embeddings=data["embeddings"]
            self.chunks = data["chunks"].tolist()
            return True
        return False