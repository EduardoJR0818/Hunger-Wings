# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# --- Configuración del Backend RAG ---
# Usaremos una función para inicializar el agente RAG (similar a Streamlit)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from Scripts.agentes.Agente_semantico import BiologySemanticAgent

CURRENT_FILE_PATH = Path(__file__).resolve()

# 2. Obtiene la ruta del directorio donde está api.py (que es la raíz del proyecto, Hunger-Wings/)
BASE_DIR = CURRENT_FILE_PATH.parent

# 3. Define la ruta del archivo .env
# Esto asume que .env está directamente en la carpeta Hunger-Wings/
DOTENV_PATH = BASE_DIR / '.env'

# Cargar las variables de entorno usando la ruta absoluta
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH)
    print(f"DEBUG: .env cargado desde: {DOTENV_PATH}")
else:
    # Esto es un caso de fallo crítico si no se encuentra
    print(f"ERROR: Archivo .env no encontrado en: {DOTENV_PATH}")


# Inicialización del Agente RAG (Global, para que solo cargue una vez)
# ====================================================================
try:
    # 1. Configuración del LLM (Esto ya está bien)
    gemini_key = os.getenv('GEMINI_API_KEY')
    gemini_llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        api_key=gemini_key,
        response_mime_type="application/json"
    )

    # 2. Configuración de Chroma
    # ------------------------------------------------------------------
    # ¡ASEGÚRATE DE QUE ESTA LÍNEA ESTÉ PRESENTE Y SIN ERRORES!
    embeddings_generator = OllamaEmbeddings(model='mxbai-embed-large')
    # ------------------------------------------------------------------
    
    DB_PATH = BASE_DIR / 'database' / 'chromadb'
    
    vector_store = Chroma(
        collection_name='biology',
        # Aquí es donde usa la variable:
        embedding_function=embeddings_generator, 
        persist_directory=str(DB_PATH)
    )

    # 3. Inicialización del Agente
    RAG_AGENT = BiologySemanticAgent(vector_store, gemini_llm)
    print("FastAPI: Agente RAG inicializado con éxito.")

except Exception as e:
    # Esto atrapará cualquier error y evitará que el servidor muera
    print(f"ERROR CRÍTICO al inicializar el agente RAG: {e}")
    RAG_AGENT = None
# ====================================================================


app = FastAPI()

# Modelo de datos que el frontend enviará
class Query(BaseModel):
    question: str
    k_chunks: int = 5 # Valor por defecto de chunks a recuperar

@app.post("/api/query_json")
def get_report(query: Query):
    """
    Punto final para recibir una pregunta y devolver el reporte y grafo JSON.
    """
    if RAG_AGENT is None:
        raise HTTPException(status_code=500, detail="El backend RAG no pudo inicializarse.")
    
    # 1. Ejecutar el agente
    json_data = RAG_AGENT.generate(query.question, query.k_chunks)

    if json_data is None:
        raise HTTPException(status_code=500, detail="La generación JSON falló. El LLM devolvió un formato inválido.")

    # 2. Devolver el diccionario de Python. FastAPI lo serializa automáticamente a JSON.
    return json_data

# CORS (Crucial para el desarrollo de Frontend/Backend)
from fastapi.middleware.cors import CORSMiddleware

origins = [
    # Permite acceso desde tu servidor de desarrollo local (ej: React/Vite/Angular)
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Puerto común de Vite
    # Agrega otros puertos si usas React, etc.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)