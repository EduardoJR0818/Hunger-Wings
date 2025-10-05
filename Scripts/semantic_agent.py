from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
# Importación para Gemini (reemplaza a ChatOpenAI)
from langchain_google_genai import ChatGoogleGenerativeAI 
import os

from dotenv import load_dotenv

# Nota: Asumo que el nombre correcto es 'PokemonSemanticAgent' basado en el 'collection_name'
from Scripts.agentes.Agente_semantico import BiologySemanticAgent

load_dotenv('.env')

# 1. Definición del LLM de Gemini
gemini_llm = ChatGoogleGenerativeAI(
    # Usamos un modelo rápido y eficiente de Gemini 2.5
    model='gemini-2.5-flash', 
    # LangChain buscará automáticamente GEMINI_API_KEY o GOOGLE_API_KEY
    # Si quieres especificarla explícitamente: api_key=os.getenv('GEMINI_API_KEY')
)

embeddings_generator = OllamaEmbeddings(
    model='mxbai-embed-large'
)

# Nota: Cambié la pregunta a una más relevante para la colección 'pokemon'
question = 'xperimental animal procedures for STS-131'

vector_store = Chroma(
    collection_name='biology',
    embedding_function=embeddings_generator,
    persist_directory='./database/chromadb'
)

# 2. Asignación del nuevo LLM al Agente
agent = BiologySemanticAgent(
    vector_store,
    gemini_llm # Usamos la instancia de Gemini
)

# El resto del flujo permanece igual
response = agent.generate_stream(question, 5)

for token in response:
    print(token.content, end='')