from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from dotenv import load_dotenv
import csv # Importamos el módulo csv necesario
from typing import Dict # Para tipado

# Importa tu clase de agente
from agentes.Agente_semantico import BiologySemanticAgent

# Cargar las variables de entorno (Asegúrate que el .env esté en la ruta correcta)
load_dotenv('Hunger-Wings\.env')

# --- RUTAS Y CONFIGURACIÓN ---
CSV_PATH = r'Hunger-Wings\database\SB_publication_PMC.csv'

# --- 0. Cargar y Mapear Metadatos del CSV ---

def load_metadata_from_csv(csv_filepath: str) -> Dict[str, str]:
    """
    Carga los títulos y links del CSV, mapeando el título a su URL.
    
    Se limpia la clave del título eliminando cualquier ruta y la extensión.
    """
    metadata_map = {}
    try:
        # Se asume que el CSV usa UTF-8 y las columnas son 'title' y 'link'
        with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
            # Usamos DictReader para manejar las columnas por nombre
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Obtenemos el título del CSV (podría contener una ruta si el CSV no es perfecto)
                title_key_full = row.get('title', '').strip()
                
                # 1. Eliminamos la ruta completa, dejando solo el nombre del archivo (ej: 'archivo.txt').
                filename_with_ext = os.path.basename(title_key_full)
                
                # 2. Eliminamos la extensión del archivo (ej: '.txt') para estandarizar la clave.
                title_key_clean, _ = os.path.splitext(filename_with_ext)
                
                link_value = row.get('link', '').strip()
                
                if title_key_clean and link_value:
                    # La clave del mapa es el título limpio (solo nombre del archivo, sin ruta ni extensión)
                    metadata_map[title_key_clean] = link_value
        print(f"DEBUG: CSV cargado exitosamente. Se mapearon {len(metadata_map)} registros de metadatos.")
        return metadata_map
    except FileNotFoundError:
        print(f"⚠️ Error: El archivo CSV no se encontró en la ruta: {csv_filepath}. Los links no estarán disponibles.")
        return {}
    except Exception as e:
        print(f"❌ Error al procesar el archivo CSV: {e}")
        return {}

ARTICLE_LINK_MAP = load_metadata_from_csv(CSV_PATH)


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
# PASAMOS EL MAPA DE LINKS AL AGENTE
agent = BiologySemanticAgent(
    vector_store,
    gemini_llm,
    article_link_map=ARTICLE_LINK_MAP # <--- NUEVO ARGUMENTO: El agente debe usar este mapa
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
    print("Posibles Causas:")
    print("1. La base de datos en './database/chromadb' está vacía o no existe.")
    print("2. El nombre de la colección ('biology') es incorrecto.")
    print("3. El modelo de embeddings ('mxbai-embed-large') es diferente al usado durante la ingesta.")
    exit()
else:
    print(f"\n✅ ÉXITO DE RECUPERACIÓN: Chroma encontró {len(test_results)} fragmentos.")
    
    # --- MODIFICACIÓN CLAVE AQUÍ ---
    # Procesamos los metadatos del primer fragmento
    first_chunk = test_results[0]
    source_full_path = first_chunk.metadata.get('source', 'N/A')
    
    # Aplicar la misma lógica de limpieza que en load_metadata_from_csv
    if source_full_path != 'N/A':
        # 1. Obtener solo el nombre del archivo con extensión
        filename_with_ext = os.path.basename(source_full_path)
        # 2. Eliminar la extensión
        source_clean, _ = os.path.splitext(filename_with_ext)
        
        # 3. ACTUALIZAR la metadata para el JSON de salida con el título limpio
        # Esto asegura que el campo 'source' en la metadata del documento (que es el que se pasa al LLM)
        # contenga solo el nombre del archivo sin la extensión.
        first_chunk.metadata['source'] = source_clean
    else:
        source_clean = 'N/A'
    
    # Imprime una parte del contenido para confirmar que es relevante, usando el título limpio
    # ¡OJO! Ahora 'source' ya contiene el título limpio gracias a la modificación anterior.
    print(f"Fragmento 1 (Source: {first_chunk.metadata.get('source')}): {first_chunk.page_content[:150]}...")
    
print("--------------------------------------------------")

# ----------------------------------------------------
## 4. Ejecución de la Generación JSON
# ----------------------------------------------------

print(f"\n--- 🤖 Generando Respuesta Estructurada con Gemini ---")

# Llamada al Agente para generar la respuesta (retorna un dict/JSON ya parseado o None)
# NOTA: Ahora el agente usará 'article_link_map' para encontrar el link de cada artículo en el resultado.
json_response_dict = agent.generate(question, k_chunks)

# ----------------------------------------------------
## 5. Impresión del Resultado Final
# ----------------------------------------------------
if json_response_dict:
    print("\n✅ Generación de JSON Exitosa. Resultado para el Frontend:")
    print(json.dumps(json_response_dict, indent=2, ensure_ascii=False))
else:
    print("\n❌ FALLO EN LA GENERACIÓN DE RESPUESTA. Esto SUCEDE porque el LLM devolvió un JSON inválido.")