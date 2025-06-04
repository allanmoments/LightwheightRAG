from document_store import DocumentStore
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from generator import Generator

class DocumentChatRAG:
    def __init__(self, doc_folder):
        self.doc_store = DocumentStore()
        self.embedder = Embedder()
        self.generator = Generator()
        
        # Load & chunk docs
        documents = self.doc_store.load_documents(doc_folder)
        self.doc_store.chunk_documents(documents)
        self.chunks = self.doc_store.get_chunks()
        
        # Embed & build index
        embeddings = self.embedder.embed_text(self.chunks)
        self.vector_store = VectorStore(dim=embeddings.shape[1])
        self.vector_store.build_index(embeddings)
        
        self.retriever = Retriever(self.embedder, self.vector_store, self.chunks)

    def chat(self, query):
        context_chunks = self.retriever.retrieve(query)
        answer = self.generator.generate_answer(query, context_chunks)
        return answer
