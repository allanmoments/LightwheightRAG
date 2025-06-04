from rag_system import DocumentChatRAG

def main():
    doc_folder = 'docs'  # Put your txt files here
    rag = DocumentChatRAG(doc_folder)
    print("Document Chat RAG ready. Type your questions (type 'exit' to quit).")

    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = rag.chat(query)
        print(f"Bot: {answer}")

if __name__ == '__main__':
    main()
