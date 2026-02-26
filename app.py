import streamlit as st
from retriever import generate_alternative_queries
from Generator import generate_answer
from cv_pipeline import CVPipeline

#CVPipeline().vector_manager.delete_collection("cv_collection")
#CVPipeline().run()  # Preprocess CVs and populate vector store at startup

st.title("CV Question Answering System")
user_query = st.text_input("Enter your question about the candidates:")
if user_query:

    relevant_chunks = generate_alternative_queries(user_query)
    answer = generate_answer(user_query, relevant_chunks)

    st.subheader("Answer:")
    st.write(answer)

    st.subheader("Relevant_Chunks:")
    relevant_chunks = [{
        "candidate_name": doc.metadata.get("candidate_name"),
        "section": doc.metadata.get("section"),
        "content": doc.page_content
    } for doc in relevant_chunks]
    st.write(relevant_chunks)