import hnswlib
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16)
        self.index.set_ef(50)  # ef should be > top_k for better recall
        self.index.set_num_threads(4)

    def build_index(self, vectors):
        self.index.add_items(vectors)

    def search(self, query_vector, top_k=3):
        labels, distances = self.index.knn_query(query_vector, k=top_k)
        return labels[0], distances[0]
