import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def process_query(query, vector_store):
    """
    Process the user query and return relevant information from the vector store.
    """
    if not query.strip():
        return "Please enter a valid query."
    
    # Use the vector store directly for similarity search instead of the retriever
    similar_docs = vector_store.similarity_search(query, k=3)
    
    if not similar_docs:
        return "No relevant information found in the document."
    
    # Join the content of similar documents
    response = "\n\n".join([doc.page_content for doc in similar_docs])
    return response

def main():
    st.title("Chat with your PDF 🦜📄")
    st.markdown("<h2 style='margin: -30px 90px;font-size:15px;'>Built by <a href ='https://github.com/Balajithirumalai05'>Balaji Thirumalai<a></h2>",unsafe_allow_html=True)
    
    # Initialize session state for vector store if it doesn't exist
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    uploaded_file = st.file_uploader("\nUpload your PDF 👇", type=["pdf"])
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Only process the PDF if we haven't already (checking session state)
            if st.session_state.vector_store is None:
                # Load PDF
                pdf_loader = PyPDFLoader("uploaded_file.pdf")
                documents = pdf_loader.load()
                
                # Extract page content
                doc_texts = [doc.page_content for doc in documents]
                
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings()
                
                # Create FAISS vector store
                st.session_state.vector_store = FAISS.from_texts(doc_texts, embeddings)
                
                st.write("Your PDF has been processed. Start chatting below!")
            
            # Chat interface
            user_query = st.text_input("Ask a question about your PDF:")
            
            if user_query:
                bot_response = process_query(user_query, st.session_state.vector_store)
                st.write("**Bot:**", bot_response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.vector_store = None  # Reset the vector store on error

if __name__ == "__main__":
    main()