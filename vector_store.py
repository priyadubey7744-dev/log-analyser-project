import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """
    Lightweight vector store using TF-IDF embeddings.

    No API key needed for embeddings — runs 100% locally.
    In production, you'd swap TF-IDF for OpenAI/Groq embeddings
    and use ChromaDB or Pinecone for storage.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)  # unigrams + bigrams for better matching
        )
        self.chunk_vectors = None
        self.chunks = []

    def build(self, chunks: list[str]):
        """
        Index all text chunks into the vector store.

        Args:
            chunks: List of text chunks from the PDF
        """
        self.chunks = chunks
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)
        print(f"[VectorStore] Indexed {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 4) -> list[str]:
        """
        Retrieve top-k most relevant chunks for a given query.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            List of the most relevant text chunks
        """
        if self.chunk_vectors is None:
            raise ValueError("Vector store is empty. Call build() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.chunk_vectors).flatten()

        # Get top-k indices sorted by similarity (descending)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices if similarities[i] > 0]
