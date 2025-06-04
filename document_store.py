import os

class DocumentStore:
    def __init__(self):
        self.chunks = []

    def load_documents(self, folder_path):
        documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        return documents

    def chunk_documents(self, documents, chunk_size=500, overlap=50):
        self.chunks = []
        for doc in documents:
            tokens = doc.split()  # simple whitespace tokenization
            start = 0
            while start < len(tokens):
                end = start + chunk_size
                chunk = ' '.join(tokens[start:end])
                self.chunks.append(chunk)
                start += chunk_size - overlap

    def get_chunks(self):
        return self.chunks
