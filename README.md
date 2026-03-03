# CVs Chat system 
This repository contains a simple implementation of a Retrieval-Augmented Generation (RAG) system designed to answer questions about candidates based on their CVs. The system consists of three main components:

1. **CVPipeline**: This component is responsible for processing CV documents, splitting them into chunks, and storing them in a vector database for efficient retrieval.

- load the CVs from the specified directory, and we support multiple formats including PDF, and DOCX.

- use **Document-Aware Chunking strategy** to split the CVs (CVs is consider semi-structured documents splited hadders and content, so it is the best strategy) ,applied it by hybrid approach between rule-based (using regex) and LLM-based methods. 

- use Qdrant as the vector database to store the chunks and their embeddings, and we follow HNSW algorithm for efficient similarity search.

2. **Retriever**: This component generates alternative queries based on the user's input and retrieves relevant chunks from the vector database.

- we follow **Multi-Query Generation strategy** ( CVs usually contain technical terms specific to each person, the common issue with this type of document is terminology mismatch. Therefore, it would be a better strategy to use this approach) to generate multiple queries from the user's input using a language model, and then we retrieve relevant chunks for each generated query, and finally we aggregate the retrieved chunks to pass them to the generator.

3. **Generator**: This component takes the retrieved chunks and generates a coherent answer to the user's question using a language model.


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