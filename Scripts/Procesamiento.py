import os
import glob
import uuid
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
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

chroma_db_path = r'Hunger-Wings\database\chromadb'

# Inicialización de la base de datos
vector_store = Chroma(
    persist_directory=chroma_db_path,
    embedding_function=embeddings_generator,
    collection_name='biology'
)

# Lógica para procesar todos los txt.

# Especificación de ruta
docs_path = r'Hunger-Wings\database\docs'

txt_files = glob.glob(os.path.join(docs_path, '*.txt'))

print(f"📄 Se encontraron {len(txt_files)} archivos .txt para procesar.")

for file_path in txt_files:
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"\nProcesando archivo: {file_name}.txt")

        # Lee el contenido del archivo
        with open(file_path, 'r', encoding='utf-8') as file:
            txt_text = file.read()

        # Divide el texto en chunks
        text_chunks = text_splitter.split_text(txt_text)
        print(f"-> Texto dividido en {len(text_chunks)} chunks.")

        # Procesamiento de cada chunk
        documents_to_add = []
        for chunk in text_chunks:
            document = Document(
                id=str(uuid.uuid4()),
                page_content=chunk,
                # Metadatos dinámicos
                metadata={
                    'name': file_name,
                    'source': file_path,
                }
            )
            documents_to_add.append(document)
        
        if documents_to_add:
            vector_store.add_documents(documents_to_add)
            print(f"-> Se agregaron {len(documents_to_add)} documentos a la base de datos.")

    except Exception as e:
        print(f"Error procesando el archivo {file_path}: {e}")
        continue

print("\nLa base de datos ha sido generada con todos los archivos.")

print("\nRealizando una búsqueda de prueba...")
search_results = vector_store.similarity_search('experimental animal procedures for STS-131', 5)

print("\nResultados de la búsqueda:")
for result in search_results:
    # Imprimime datos del archivo
    print(f"Fuente: {result.metadata.get('name', 'N/A')}.txt")
    print(f"Contenido: {result.page_content[:200]}...\n")


"""
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

resultados = vector_store.similarity_search('experimental animal procedures for STS-131', 5)

print("-" * 70)
print("🔎 Resultados de la Búsqueda de Similitud para 'experimental animal procedures for STS-131' 🔎")
print("-" * 70)

# Iteramos sobre cada documento encontrado
for i, doc in enumerate(resultados):
    # Extraemos el texto del documento
    contenido_encontrado = doc.page_content
    
    # Intentamos obtener la fuente/metadata (normalmente la ruta del archivo)
    # Usamos .get() para evitar errores si la metadata no contiene 'source'
    fuente = doc.metadata.get('source', 'Fuente no especificada')
    
    # Imprimimos el resultado con formato
    print(f"\n--- Documento Relevante #{i+1} ---")
    print(f"📄 Fuente: {fuente}")
    
    # Imprimimos los primeros 700 caracteres del texto para concisión
    print(f"📝 Texto Relevante:\n{contenido_encontrado[:700]}...") 
    print("-" * 30)
    """