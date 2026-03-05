from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from cv_pipeline import CVPipeline
import os
from dotenv import load_dotenv
load_dotenv()


def generate_alternative_queries(query, vectorstore, docs):

    # Step 1: generate multiple queries using LLM
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = os.getenv("GOOGLE_API_KEY3"), temperature=0.5)
   
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant that generates alternative phrasings for a given question about candidate CVs.
        Your task is to create 3 different ways to ask the same question, which can help in retrieving more relevant information from CVs.
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

    

    all_docs = []
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    hybrid_retriever = EnsembleRetriever( retrievers=[bm25_retriever, semantic_retriever], weights=[0.4, 0.6])

    #docs = hybrid_retriever.get_relevant_documents(query)

    for q in queries:
        retrieved = hybrid_retriever.invoke(q)
        all_docs.extend(retrieved)

    #----------------------------------------------------------------------
    # Step 3: remove duplicates

    unique_docs = {doc.page_content: doc for doc in all_docs}
    final_docs = list(unique_docs.values())

    return final_docs

