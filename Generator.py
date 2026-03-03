from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from collections import defaultdict
import os
from dotenv import load_dotenv
load_dotenv()

def generate_answer( query, relevant_chunks):

    # chunks that have same candidate_name ----> collect them in same document ------->
    # to prevent model from seeing same candidate multiple times with different sections. 
    merged = defaultdict(str)
    for doc in relevant_chunks:
       sur =  doc.metadata["candidate_name"]
       merged[sur] += doc.page_content + "\n"


    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="\n".join(["""
            You are an HR assistant answering questions about job candidates.

            You MUST base your answer ONLY on the provided context.

            Each context chunk contains:
            - candidate_name
            - section
            - content

            Instructions:

            1) Identify ALL candidate(s) which match the question.
            2) ALWAYS mention the candidate's real name (candidate_name).
            3) For each candidate, clearly mention:
            - The candidate's name
            - The section where the information was found
            - A short explanation
            4) If multiple candidates match, Mention all of them separately.
            5) If no candidate matches, say:
            "No candidates were found."
           
            6) Do NOT include information that is not in the context.
            
            7) Respond as if speaking to a recruiter.
            8) IF the question is not relevant to the candidates' information except Greetings like "Hello", "Hi", "Good Morning", 
               say:"The question is not relevant to the candidates' information."
                but If the question is a related to the candidates' information but is not answerable based on the provided context, say "The answer is not available in the provided CVs."
                            
            9) If asked about years of experience, look for explicit mentions of years in the content, if not mentioned infer years of experience from job durations or dates from experience section.
       
           10) If the question ask you to response in a specific format like "Answer in one line", "Answer in bullet points", "Answer in a table" YOU MUST follow the instruction and format your answer accordingly, if not specified answer in a concise paragraph.            
           11) IF the question is asking for an imaginary positions that are not common as a job title search firstly about this position to check if this position is imaginary or not
                IF is imaginary , say "It is an Imaginary Position." Do NOT hallucinate or infer any answer.
            
            Format your answer like this:

            - Candidate Name:
            Section:
            Explanation:

            Context:
            {context}

            Question:
            {query}

            Answer:
            """
        ])
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = os.getenv("GOOGLE_API_KEY"), temperature=0.2)
 
    parser = StrOutputParser()
    rag_chain = prompt | llm | parser

    answer = rag_chain.invoke({"context": merged, "query": query})

    return answer