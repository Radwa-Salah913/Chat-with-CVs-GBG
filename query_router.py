from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

def router(query):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key = os.getenv("GOOGLE_API_KEY2"))
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are a Query Routing Assistant for a multi-document question answering system.

        The user uploads exactly 5 documents (e.g., CVs or benefit-related documents).
        Your task is to classify the user's question into ONE and ONLY ONE of the following categories.

        Available Categories:

        1. invalid_role
        → The question asks about a job role or concept that is not a real or recognized job title.
        - If the question contains a fictional or meaningless role.
        - whether the user's question is NOT valid based on real job roles and CV information.
        Examples of VALID job roles:
                Data Scientist
                Machine Learning Engineer
                AI Engineer
                Backend Developer
                Frontend Developer
                Data Analyst
                Software Engineer

        2. factual_lookup
        → The question asks for a specific piece of information directly stated in the documents.
        Example: "What is Ahmed's GPA?"

        3. filtering
        → The question asks to find documents that match a condition.
        Example: "Who knows Python?"

        4. comparison
        → The question compares two or more documents.
        Example: "Compare the education of the five candidates."

        5. ranking_selection
        → The question asks to choose the best candidate based on criteria.
        Example: "Who is the best fit for a Data Scientist role?"

        6. counting
        → The question asks for a count or number.
        Example: "How many candidates have internships?"

        7. summarization
        → The question asks to summarize one or more documents.
        Example: "Summarize candidate 3."

        8. analytical_reasoning
        → The question requires reasoning, inference, or interpretation beyond direct facts.
        Example: "Who seems more research-oriented?"

        9. out_of_scope
        → The question is unrelated to the uploaded documents.
        Example: "What is the capital of France?"

        -----------------------------------------------------

        Strict Instructions:

        - You MUST return only one category.
        - Do NOT explain your reasoning.
        - Do NOT answer the question.
        - Do NOT return anything except valid JSON.
        - If the question mentions a job role that is unrealistic, fictional, or not a known role, choose "invalid_role".
        - If the question is unrelated to the uploaded documents, choose "out_of_scope".

        -----------------------------------------------------

        Return format:

        
        "category": "<one_of_the_categories_above>"
        

        User Question:
        {question}
       """
    )
    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"question": query})

    return response["category"]

