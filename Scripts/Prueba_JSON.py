from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 
import os
import json
from dotenv import load_dotenv

# Importa tu clase de agente
from agentes.Agente_semantico import BiologySemanticAgent 

# Cargar las variables de entorno (Asegúrate que el .env esté en la ruta correcta)
load_dotenv('Hunger-Wings\.env') 

# ----------------------------------------------------
## 1. Configuración del LLM
# ----------------------------------------------------

# VERIFICACIÓN DE LA API KEY (Evita DefaultCredentialsError)
gemini_key = os.getenv('GEMINI_API_KEY')
if not gemini_key:
    raise ValueError(
        "ERROR: La variable GEMINI_API_KEY no se encontró después de cargar el .env. "
        "¡Revisa tu archivo y nombre de variable!"
    )
print(f"DEBUG: GEMINI_API_KEY cargada correctamente (longitud: {len(gemini_key)} caracteres)")

# Inicialización de Gemini LLM (response_mime_type es esencial para el JSON)
gemini_llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=gemini_key,
    response_mime_type="application/json" 
)

# ----------------------------------------------------
## 2. Configuración de RAG
# ----------------------------------------------------

# Configuración de Embeddings (Debe coincidir con la ingesta)
embeddings_generator = OllamaEmbeddings(
    model='mxbai-embed-large'
)

# Configuración de Chroma Vector Store (Asegúrate de la ruta y collection_name)
vector_store = Chroma(
    collection_name='biology',
    embedding_function=embeddings_generator,
    persist_directory='Hunger-Wings\database\chromadb'
)

# Instanciación del Agente Biológico
agent = BiologySemanticAgent(
    vector_store,
    gemini_llm 
)

# ----------------------------------------------------
## 3. Prueba de Recuperación (Diagnóstico del Problema)
# ----------------------------------------------------
question = 'Explica los procedimientos experimentales con animales para la misión STS-131'
k_chunks = 5

print(f"\n--- 🧪 Probando Recuperación Cruda de Chroma para '{question[:50]}...' ---")

# Ejecuta la recuperación directamente sobre la base de datos
test_results = vector_store.similarity_search(question, k_chunks)

if not test_results:
    print("\n🚨 ERROR CRÍTICO DE RECUPERACIÓN: Chroma no devolvió ningún documento para la colección 'biology'.")
    print("   Posibles Causas:")
    print("   1. La base de datos en './database/chromadb' está vacía o no existe.")
    print("   2. El nombre de la colección ('biology') es incorrecto.")
    print("   3. El modelo de embeddings ('mxbai-embed-large') es diferente al usado durante la ingesta.")
    exit()
else:
    print(f"\n✅ ÉXITO DE RECUPERACIÓN: Chroma encontró {len(test_results)} fragmentos.")
    # Imprime una parte del contenido para confirmar que es relevante
    print(f"   Fragmento 1 (Source: {test_results[0].metadata.get('source', 'N/A')}): {test_results[0].page_content[:150]}...")
    
print("--------------------------------------------------")

# ----------------------------------------------------
## 4. Ejecución de la Generación JSON
# ----------------------------------------------------

print(f"\n--- 🤖 Generando Respuesta Estructurada con Gemini ---")

# Llamada al Agente para generar la respuesta (retorna un dict/JSON ya parseado o None)
json_response_dict = agent.generate(question, k_chunks)

# ----------------------------------------------------
## 5. Impresión del Resultado Final
# ----------------------------------------------------
if json_response_dict:
    print("\n✅ Generación de JSON Exitosa. Resultado para el Frontend:")
    print(json.dumps(json_response_dict, indent=2, ensure_ascii=False))
else:
    print("\n❌ FALLO EN LA GENERACIÓN DE RESPUESTA. Esto SUCEDE porque el LLM devolvió un JSON inválido.")