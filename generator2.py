from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from collections import defaultdict
import os
from dotenv import load_dotenv
load_dotenv()

def generate_final_answer(query, relevant_chunks, query_instructions):
    # chunks that have same candidate_name ----> collect them in same document ------->
    # to prevent model from seeing same candidate multiple times with different sections. 
    merged = defaultdict(str)
    for doc in relevant_chunks:
       sur =  doc.metadata["candidate_name"]
       merged[sur] += doc.page_content + "\n"

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = os.getenv("GOOGLE_API_KEY"), temperature=0.1)
    parser = StrOutputParser()

    prompt = PromptTemplate(
        input_variables=["context", "question", "instructions"],
        template="""
            You are an HR assistant answering questions about job candidates.

            You MUST base your answer ONLY on the provided context.  
            
            GLOBAL RULES:

            - Never generate information not explicitly present.
            - If multiple candidates match, Mention all of them separately ,but NOT repeat them .
            - Never normalize job titles.
            - If the question ask you to response in a specific format like "Answer in one line", "Answer in bullet points", "Answer in a table" YOU MUST follow the instruction and format your answer accordingly, if not specified answer in a concise paragraph.  
            - For each candidate, clearly mention:
                     1- The candidate's name
                     2- The section where the information was found
                     3- A short explanation
                     
            TASK-SPECIFIC INSTRUCTIONS:
            {instructions}

            Context:
            {context}

            User Question:
            {question}  

            Answer:
            ```  

            """
    )
    chain = prompt | llm | parser
    response = chain.invoke({"question":query, "context": merged, "instructions":query_instructions})

    return response
