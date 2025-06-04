class Retriever:
    def __init__(self, embedder, vector_store, chunks):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks = chunks

    def retrieve(self, query, top_k=3):
        q_vec = self.embedder.embed_text([query])
        labels, _ = self.vector_store.search(q_vec, top_k)
        results = [self.chunks[i] for i in labels]
        return results
