import os
import re
import ntpath
import hashlib
from typing import List, Optional
from dotenv import load_dotenv
from collections import defaultdict
from pydantic import BaseModel, Field
import pymupdf4llm  

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff


load_dotenv()
############################################################################################

class CVLoader:

    def __init__(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.dir_path = os.path.join(root_path,"assets","temp_uploads")

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

    # -------------------------------------------------------------
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
            template=" ".join(["""
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
        
        print(f"Regex splitting failed for document: {document.metadata['source']}, using LLM splitting.")

        return self.llm_structure_split(
            document.page_content,
            document.metadata["source"]
        )

###############################################################################################
class CVSpliter:
    """Splits CV documents into sections using markdown header detection."""
    
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "candidate"),
            ("##", "section"),
        ]
    
    def _is_heading(self, line: str) -> bool:
        """
        Determines if a line is a heading based on heuristic rules.
        
        Args:
            line: The line to check
            
        Returns:
            True if the line appears to be a heading, False otherwise
        """
        stripped = line.strip()
        
        # Empty line
        if not stripped:
            return False
        
        # Too long to be a header
        if len(stripped) > 60:
            return False
        
        # Ends with period? Usually a paragraph
        if stripped.endswith('.'):
            return False
        
        # Too many words? Usually not a header
        if len(stripped.split()) > 6:
            return False
        
        # Too many digits? Usually not a header
        if sum(c.isdigit() for c in stripped) > 3:
            return False
        
        # Check for Title Case or ALL CAPS
        if stripped.istitle() or stripped.isupper():
            return True
        
        return False
    
    def _auto_convert_to_markdown(self, text: str) -> str:
        """
        Automatically converts text to markdown format with proper headers.
        
        Args:
            text: The raw text to convert
            
        Returns:
            Markdown formatted text
        """
        lines = text.split("\n")
        new_lines = []
        
        # Make first non-empty line a main header
        first_line_added = False

        for line in lines:
            if not first_line_added and line.strip():
                new_lines.append(f"# {line.strip()}")
                first_line_added = True
                continue
            
            if self._is_heading(line):
                new_lines.append(f"\n## {line.strip()}\n")
            else:
                new_lines.append(line)

        return "\n".join(new_lines)
    
    def split(self, file_path: str) -> List[Document]:
        """
        Converts a CV file to markdown and splits it by headers.
        
        Args:
            file_path: Path to the CV document (PDF or DOCX)
            
        Returns:
            List of Document objects, each representing a section
        """
        # Convert document to markdown
        markdown_text = pymupdf4llm.to_markdown(file_path)
        
        # Auto-convert to proper markdown format
        markdown_text = self._auto_convert_to_markdown(markdown_text)
        
        # Split by headers
        splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)
        docs = splitter.split_text(markdown_text)
        
        # Add metadata
        candidate_name = os.path.splitext(ntpath.basename(file_path))[0]
        for doc in docs:
            doc.metadata["source"] = file_path
            doc.metadata["candidate_name"] = candidate_name
        
        return docs


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
        # generate unique ids for each chunk based on content + source to avoid duplicates.
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

    def __init__(self,collection_name: str):
        self.loader = CVLoader()
        self.chunker = CVSpliter()
        self.vector_manager = VectorStoreManager(collection_name)

    def run(self):
        docs = self.loader.load_documents()

        final_chunks = []
        for doc in docs:
            chunks = self.chunker.hybrid_chunk(doc)
            final_chunks.extend(chunks)

        self.vector_manager.add_documents(final_chunks)
        
        print(f"Processed {len(docs)} documents into {len(final_chunks)} chunks and added to vector store.\n\n")
        return final_chunks

   
