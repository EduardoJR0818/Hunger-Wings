import json
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import VectorStore
# Se asume que _PROMP y _JSON_SCHEMA están definidos en este archivo o se importan.
# Para este ejemplo, los definiremos como placeholders.

# ----------------------------------------------------
# PLACEHOLDERS: Sustituye esto con tus definiciones reales
# ----------------------------------------------------
_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reporte": {
            "type": "object",
            "properties": {
                "resumen": {"type": "string", "description": "Resumen conciso del hallazgo."},
                "hallazgos": {"type": "array", "items": {"type": "string"}, "description": "Lista de puntos clave."},
            }
        },
        "grafo": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "palabra": {"type": "string"},
                    "articulos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "titulo": {"type": "string"},
                                "link": {"type": "string", "description": "URL de la fuente original."},
                            }
                        }
                    },
                    "relaciones": {"type": "array", "items": {"type": "string"}},
                }
            }
        }
    }
}

_PROMPT = """
Eres un Agente Semántico de Biología Espacial. Tu tarea es responder a la pregunta del usuario utilizando solo el contexto proporcionado.
Genera una respuesta en formato JSON que cumpla exactamente con el esquema proporcionado.

Instrucciones Clave:
1. Analiza el 'contexto_filtrado' para responder la 'pregunta_usuario'.
2. El campo 'reporte' debe contener el resumen de la respuesta.
3. El campo 'grafo' debe contener los conceptos clave extraídos del contexto, sus artículos de origen, y las relaciones entre ellos.
4. **IMPORTANTE:** Para cada artículo en el campo 'grafo', utiliza el 'titulo' y 'link' exactos proporcionados en la metadato de los documentos fuente.
5. El output DEBE ser un JSON válido.

Contexto Filtrado (Documentos Fuente y Metadatos):
---
{contexto_filtrado}
---

Pregunta del Usuario: {question}
"""
# ----------------------------------------------------
# FIN DE PLACEHOLDERS
# ----------------------------------------------------


class BiologySemanticAgent:
    """
    Agente que realiza RAG sobre una base de datos vectorial
    y estructura la respuesta en un formato JSON para visualización de grafo.
    """
    # 1. ACTUALIZACIÓN DEL CONSTRUCTOR PARA ACEPTAR article_link_map
    def __init__(self, vector_store: VectorStore, llm, article_link_map: Dict[str, str]):
        self.vector_store = vector_store
        self.llm = llm
        self.article_link_map = article_link_map  # Almacenamos el mapa de links
        
        # Inicializa el parser de JSON
        self.json_parser = JsonOutputParser()
        self.prompt_template = _PROMPT
        
        # Cadena de Generación: Prompt -> LLM (forzado JSON) -> Parser (a dict de Python)
        # La cadena se construye en generate para incluir el contexto dinámicamente.


    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepara los documentos para el prompt, enriqueciendo los metadatos con el 'link' del CSV.
        
        Se asume que la metadato 'source' de LangChain contiene el TÍTULO exacto del artículo
        usado como clave en ARTICLE_LINK_MAP.
        """
        prepared_data = []
        
        for doc in documents:
            title = doc.metadata.get('source', 'Título Desconocido')
            # 2. USAMOS EL MAPA PARA BUSCAR EL LINK
            link = self.article_link_map.get(title, 'Link No Encontrado') 
            
            # Crea un diccionario simple con los datos relevantes para el LLM
            # Incluimos el link para que el LLM lo use al llenar el campo 'articulos' del JSON.
            prepared_data.append({
                "source_title": title,
                "source_link": link,
                "snippet": doc.page_content[:200] + "...", # Recortar para no saturar el prompt
                "full_page_content": doc.page_content
            })

        # Serializa la lista de diccionarios para inyectarla en el prompt
        return json.dumps(prepared_data, indent=2, ensure_ascii=False)


    def generate(self, question: str, k_chunks: int) -> Dict[str, Any] | None:
        """
        Realiza la recuperación y la generación de la respuesta estructurada.
        """
        # 1. Recuperación
        retrieved_docs = self.vector_store.similarity_search(question, k_chunks)
        
        if not retrieved_docs:
            print("No se recuperaron documentos para generar la respuesta.")
            return None

        # 2. Preparación del Contexto (con links inyectados)
        contexto_json = self._prepare_context(retrieved_docs)

        # 3. Creación del Prompt Final
        prompt_with_context = self.prompt_template.format(
            contexto_filtrado=contexto_json,
            question=question
        )
        
        # 4. Configuración y Ejecución de la Cadena
        try:
            # Usamos el LLM directamente ya que la cadena se complica con el formato
            # El LLM ya está inicializado para devolver JSON según el Canvas (response_mime_type)
            response = self.llm.invoke(
                [("user", prompt_with_context)],
                config={"response_schema": _JSON_SCHEMA}
            )
            
            # Intenta parsear la respuesta de texto (que debería ser JSON)
            json_text = response.content.strip()
            return json.loads(json_text)

        except Exception as e:
            print(f"Error durante la generación o parseo del JSON: {e}")
            # print(f"Respuesta del LLM (si disponible): {response.content}")
            return None
