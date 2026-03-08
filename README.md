# CVs Chat system 
This repository contains a simple implementation of a Retrieval-Augmented Generation (RAG) system designed to answer questions about candidates based on their CVs. The system consists of main components:

1. **CVPipeline**: This component is responsible for processing CV documents, splitting them into chunks, and storing them in a vector database for efficient retrieval.

- load the CVs from the specified directory, and we support multiple formats including PDF, and DOCX.

- use **Document-Aware Chunking strategy** to split the CVs (CVs is consider semi-structured documents splited hadders and content, so it is the best strategy) 

- we can applied it by hybrid approach between rule-based (using regex) and LLM-based methods, but there are some limitations to this approach like the variability in CV Headers so we can't determine a fixed set of headers to split the CVs + if there is a word like "experience" in the content of the CV not as a header it will cause a problem. ------> it try it **CVChunker class** in CV_pipeline.py file.

- we can Splits CV documents into sections using markdown header detection **MarkdownHeaderTextSplitter** from langchain, but it is not good with most of the CVs and results with it are not good. ------> it try it **CVSpliter class** in CV_pipeline.py file.

- the best approch that return the most good results is to use **partition_pdf** library from **Unstructured**, it detect each line in the file if is a **title** or not ,but it is alone is not enough beccause there are a limitation but I try to handle it as much as possible by using some rules to determine if the line is a title or not, and it return good results with most of the CVs. ------> it try it **CVLaderandSpliter class** in CV_pipeline.py file.

- use **Qdrant** as the vector database to store the chunks and their embeddings, and we follow **HNSW** algorithm for efficient similarity search.

2. **Retriever**: This component generates alternative queries based on the user's input and retrieves relevant chunks from the vector database.

- we follow **Multi-Query Generation strategy** ( CVs usually contain technical terms specific to each person, the common issue with this type of document is terminology mismatch. Therefore, it would be a better strategy to use this approach) to generate multiple queries from the user's input using a language model.

- I applied **Hybrid-RAG** Concept to retrieve relevant chunks , which means that I use both semantic search and keyword-based search, retrieve **15 chunk** from each of them and then combine the results and remove duplicates.

- applied **Re-ranking** to the retrieved chunks using a language model to ensure that the most relevant information is prioritized in the final answer generation.

3. **Generator**: This component takes the retrieved chunks and generates a coherent answer to the user's question using a language model.

4. **query_router**: This component is responsible for routing the user's query to the appropriate retriever based on the content of the query. It uses a LLm to analyze the query and determine which category it belong to (e.g., **factual_lookup**, **filtering**, **comparison**, **ranking_selection**, **counting**, **out_of_scope**, etc.) or the question contains a fictional or meaningless role.

5. **Instructions.json**: This file inside **assets** directory contains the instructions for the query router, which helps it determine how to route different types of queries.

6. **generator2** : this component take the user query, it's category (which is determined by the query router) and the retrieved chunks, and generate a coherent answer to the user's question using a language model, but it also take into consideration the category of the question to generate a more accurate answer.

- YOU CAN TRY BOTH GENERATOR AND GENERATOR2 TO SEE THE DIFFERENCE IN THE ANSWERS, BY CHANGING THE GENERATOR IN THE **app.py** FILE.

## Setup Instructions
1. create a new conda environment and activate it:
```bash
conda create -n env_name python=3.11 -y
conda activate env_name
```
## Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Set up environment variables:
- Create a `.env` file in the root directory of the project and add your Google API
```bash
cp .env.example .env
```
- Edit the `.env` file to include your **Google API key** , **Qdrant API key** and its **URL**.

## Usage
To run the application, use the following command:
```bash
streamlit run app.py
```
- This will start the Streamlit application, and you can interact with it through your web browser. 

- You must upload exactly 5 CVs in the specified formats (PDF or DOCX) to the application. Once the CVs are processed and stored  in the vector database, you can enter questions about the candidates. The system will retrieve relevant information from the CVs and generate answers accordingly.

- each time you upload new CVs, the system will create a new collection with new collection name in the vector database and replace it with the new CVs, so make sure to upload all the CVs you want to use at once.