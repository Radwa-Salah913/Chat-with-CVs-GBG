from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from cv_pipeline import CVPipeline
import os
from dotenv import load_dotenv
load_dotenv()


def generate_alternative_queries(query):

    # Step 1: generate multiple queries using LLM
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = os.getenv("GOOGLE_API_KEY"), temperature=0.7)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant that generates alternative phrasings for a given question about candidate CVs.
        Your task is to create 4 different ways to ask the same question, which can help in retrieving more relevant information from CVs.
        DO NOT change the meaning of the question, only rephrase it.
        DO NOT repeat the same question, each question must be significantly different in structure or wording.

        {question}
        Return each on a new line.
        """,
        input_variables=["question"]
    )

    chain = prompt | llm | StrOutputParser()

    queries = chain.invoke({"question": query}).split("\n")

    # ---------------------------------------------------------------------
    # Step 2: retrieve top chunks for each query

    cv_pipeline = CVPipeline()
    vectorstore = cv_pipeline.vector_manager.get_vectorstore()

    all_docs = []
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    for q in queries:
        retrieved = retriever.invoke(q)
        all_docs.extend(retrieved)

    #----------------------------------------------------------------------
    # Step 3: remove duplicates

    unique_docs = {doc.page_content: doc for doc in all_docs}
    final_docs = list(unique_docs.values())

    return final_docs

