import os
import re
import ntpath
import hashlib
from typing import List, Optional
from dotenv import load_dotenv
from collections import defaultdict
from pydantic import BaseModel, Field

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff


load_dotenv()
############################################################################################

class CVLoader:

    def __init__(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.dir_path = os.path.join(root_path,"assets","CVs")

    def _load_pdfs(self):
        loader = DirectoryLoader(
            self.dir_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()

    def _load_docx(self):
        loader = DirectoryLoader(
            self.dir_path,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader
        )
        return loader.load()
    
    # pdf loader load each page as a single document ----> so we need merge pages that in same document.
    def _merge_pdf_pages(self, pdf_docs):
        merged = defaultdict(str)

        # pages that have same source ----> collect them in same document (document is a key in the dic)
        for doc in pdf_docs:
            source = doc.metadata["source"]
            merged[source] += doc.page_content + "\n"

        return [
            Document(
                page_content=content,
                metadata={"source": source} # source is ----> the full path of the document
            )
            for source, content in merged.items()
        ]
    
    def load_documents(self) -> List[Document]:
        pdf_docs = self._load_pdfs()
        docx_docs = self._load_docx()
        merged_pdf_docs = self._merge_pdf_pages(pdf_docs)
        return merged_pdf_docs + docx_docs



###########################################################################################

# Most common CV sections headers.
COMMON_HEADERS = [
    "Education",
    "Experience",
    "Work Experience",
    "Projects",
    "Skills",
    "Certifications",
    "Summary",
    "Profile"
]

class Section_format(BaseModel):
    section_title: str
    content: str


class CVSections(BaseModel):
    section: List[Section_format] = Field(..., description="List of sections in the CV, each section has a title and content")


class CVChunker:

    def __init__(self):
        self.model = ChatOpenAI(
            model="openai/gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.2,
            max_completion_tokens=1024
        )

    # --------------------------------
    def regex_structure_split(self, text: str, source: str) -> Optional[List[Document]]:

        pattern = r"(" + "|".join(COMMON_HEADERS) + r")"
        splits = re.split(pattern, text, flags=re.IGNORECASE)

        if len(splits) < 3:
            return None

        documents = []
        candidate_name = os.path.splitext(ntpath.basename(source))[0]

        for i in range(1, len(splits), 2):
            header = splits[i]
            content = splits[i + 1]

            documents.append(
                Document(
                    page_content=header + "\n" + content,
                    metadata={
                        "source": source,
                        "section": header,
                        "candidate_name": candidate_name
                    }
                )
            )

        return documents

   # ---------------------------------------------
    def llm_structure_split(self, text: str, source: str) -> List[Document]:

        parser = PydanticOutputParser(pydantic_object=CVSections)

        prompt = PromptTemplate(
            template="\n".join(["""
                    Extract CV sections.
                    Return JSON only.

                    Rules:
                    - Do NOT add empty objects.
                    - Each section must contain:
                    - section_title (string)
                    - content (string)
                    - Do NOT include empty sections.
                    - Do NOT include trailing commas

                    {format_instructions}

                    CV:
                    {cv_text}"""
            ]),
            input_variables=["cv_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.model | parser
        response = chain.invoke({"cv_text": text})

        documents = []
        candidate_name = os.path.splitext(ntpath.basename(source))[0]

        for sec in response.section:
            documents.append(
                Document(
                    page_content=f"{sec.section_title}\n{sec.content}",
                    metadata={
                        "source": source,
                        "section": sec.section_title,
                        "candidate_name": candidate_name
                    }
                )
            )

        return documents

    # ---------------------------------------------
    def hybrid_chunk(self, document: Document) -> List[Document]:
        regex_docs = self.regex_structure_split(
            document.page_content,
            document.metadata["source"]
        )

        if regex_docs:
            return regex_docs

        return self.llm_structure_split(
            document.page_content,
            document.metadata["source"]
        )



#################################################################################################

class VectorStoreManager:

    def __init__(self, collection_name="cv_collection"):
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True}
        )

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        self._ensure_collection()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

    def _ensure_collection(self):

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=7,
                    ef_construct=100
                )
            )

    def add_documents(self, documents: List[Document]):
        ids = []

        for doc in documents:
            unique_string = doc.page_content + doc.metadata["source"]
            chunk_id = hashlib.md5(unique_string.encode()).hexdigest()
            ids.append(chunk_id)

        self.vectorstore.add_documents(documents, ids=ids)


    def delete_collection(self,collection_name):
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' has been deleted.")
        else:
            print(f"Collection '{collection_name}' does not exist.")


    def get_vectorstore(self):
        return self.vectorstore


###################################################################################################
class CVPipeline:

    def __init__(self):
        self.loader = CVLoader()
        self.chunker = CVChunker()
        self.vector_manager = VectorStoreManager()

    def run(self):
        docs = self.loader.load_documents()

        final_chunks = []
        for doc in docs:
            chunks = self.chunker.hybrid_chunk(doc)
            final_chunks.extend(chunks)

        self.vector_manager.add_documents(final_chunks)
        
        print(f"Processed {len(docs)} documents into {len(final_chunks)} chunks and added to vector store.\n\n")
        return final_chunks


