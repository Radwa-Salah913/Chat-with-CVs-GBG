from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

def generate_answer( query, relevant_chunks):
    context = "\n\n".join(
        f"Candidate: {doc.metadata.get('candidate_name')}\nSection: {doc.metadata.get('section')}\nContent:{doc.page_content}"
        for doc in relevant_chunks
    )
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

1) Identify which candidate(s) match the question.
2) ALWAYS mention the candidate's real name (candidate_name).
3) For each candidate, clearly mention:
   - The candidate's name
   - The section where the information was found
   - A short explanation
4) If multiple candidates match, group the answer by candidate.
5) If no candidate matches, say:
   "No candidates were found with that skill."
6) Do NOT summarize sections without linking them to a candidate.
7) Do NOT include information that is not in the context.
8) Do NOT refer to "the document" or "the context".
9) Respond as if speaking to a recruiter.

Format your answer like this:

- Candidate Name:
  Section:
  Explanation:

Context:
{context}

Question:
{query}

Answer:
"""])
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = os.getenv("GOOGLE_API_KEY"))
    parser = StrOutputParser()
    rag_chain = prompt | llm | parser

    answer = rag_chain.invoke({"context": context, "query": query})

    return answer