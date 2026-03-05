import streamlit as st
import os
import shutil
import numpy as np
from cv_pipeline import CVPipeline
from retriever import generate_alternative_queries
from generator2 import generate_final_answer
from query_router import router
import json

st.title("CV Question Answering System")

# -------------------------------
# Upload Section
# -------------------------------

uploaded_files = st.file_uploader(
    "Upload exactly 5 CVs",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if "processed" not in st.session_state:
    st.session_state.processed = False
if uploaded_files and not st.session_state.processed:

    if len(uploaded_files) != 5:
        st.error("Please upload exactly 5 CV files.")

    else:
        root_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(root_path, "assets", "temp_uploads")

        # delete old temp files if exist and create new temp directory for current session uploads
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # save uploaded files to temp directory
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        # generate a unique collection name for each session to prevent conflicts between different users.
        collection_name = "CV_Collection_" + str(np.random.randint(0, 100000))
        print("Collection Name:", collection_name,"\n\n")

        
        with st.spinner("Processing CVs..."):
            pipeline = CVPipeline(collection_name)
            docs = pipeline.run()  # process + store
            vectorstore = pipeline.vector_manager.get_vectorstore()

        
        st.session_state.processed = True
        st.session_state.collection_name = collection_name
        st.session_state.docs = docs
        st.session_state.vectorstore = vectorstore

        st.success("CVs processed and stored successfully!")

# --------------------------------
#  Question Section
# ---------------------------------


if st.session_state.processed:
    path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(path, "assets", "instructions.json")
    with open(file_path, "r") as f:
        config = json.load(f)

    user_query = st.text_input("Enter your question about the candidates:")

    if user_query:

        relevant_chunks = generate_alternative_queries(user_query, st.session_state.vectorstore, st.session_state.docs)
        query_category = router(user_query)

        if query_category == "invalid_role":
            answer = "Please check if your Question is a actually real question"
        else:
            query_instructions = config[query_category]["instructions"]
            answer = generate_final_answer(user_query, relevant_chunks, query_instructions)
        
        print("Query Category : ",query_category)
        print("NUmber of relevant chunks :", len(relevant_chunks),"\n")
        
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Relevant CV Sections:")
        for i, chunk in enumerate(relevant_chunks):
            st.markdown(f"**Candidate Name:** {chunk.metadata['candidate_name']}")
            st.markdown(f"**Section:** {chunk.metadata['section']}")
            st.markdown(f"**Content:** {chunk.page_content}")
            st.markdown("---")