from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import uuid
from langchain_community.document_loaders import PyPDFLoader

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=510,
    chunk_overlap=50,
    length_function=len,
    separators=['\n', '.', '\n\n']
)

embeddings_generator = OllamaEmbeddings(
    model='mxbai-embed-large:latest'
)

chroma_db_path = 'Hunger-Wings\database\chromadb'

vector_store = Chroma(
    persist_directory=chroma_db_path,
    embedding_function=embeddings_generator,
    collection_name='pokemon'
)

txt_path = 'Hunger-Wings\database\docs\Microgravity Induces Pelvic Bone Loss through Osteoclastic Activity.txt'
txt_text = ''

with open(txt_path, 'r', encoding='utf-8') as file:
    txt_text = file.read()

print(len(txt_text))

text_chunks = text_splitter.split_text(txt_text)
print(len(text_chunks))

for chunk in text_chunks:
    document = Document(
        id=str(uuid.uuid4()),
        page_content=chunk,
        metadata={
            'name': 'Microgravity Induces Pelvic Bone Loss through Osteoclastic Activity',
            'source': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3630201/',
            }
    )

    vector_store.add_documents([document])

vector_store.similarity_search('experimental animal procedures for STS-131', 5)