from langchain.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from typing import Any, Dict, List

_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reporte": {
            "type": "object",
            "description": "El resumen ejecutivo y hallazgos clave de la respuesta.",
            "properties": {
                "resumen": {"type": "string", "description": "Un resumen conciso y natural de la respuesta generada."},
                "hallazgos": {
                    "type": "array",
                    "description": "Una lista de 3 a 5 puntos clave (hallazgos) extraídos del contexto.",
                    "items": {"type": "string"}
                }
            },
            "required": ["resumen", "hallazgos"]
        },
        "grafo": {
            "type": "array",
            "description": "Lista de 3 a 5 conceptos clave para construir el grafo de conocimiento.",
            "items": {
                "type": "object",
                "properties": {
                    "palabra": {"type": "string", "description": "La palabra o concepto más frecuente en los documentos."},
                    "articulos": {
                        "type": "array",
                        "description": "Información de los documentos de LangChain relacionados con esta palabra clave.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "titulo": {"type": "string", "description": "El título o nombre del documento (metadata 'name')."},
                                "link": {"type": "string", "description": "La URL o ruta de origen del documento (metadata 'source')."}
                            },
                            "required": ["titulo", "link"]
                        }
                    },
                    "relaciones": {
                        "type": "array",
                        "description": "Una lista de 3 palabras clave relacionadas que podrían formar un nodo en el grafo.",
                        "items": {
                            "type": "string",  # <--- ESTE ES EL CAMPO CORREGIDO
                            "description": "Una palabra clave o concepto relacionado."
                        }
                    }
                },
                "required": ["palabra", "articulos", "relaciones"]
            }
        }
    },
    "required": ["reporte", "grafo"]
}


# =======================================================
# DEFINICIÓN DEL PROMPT
# =======================================================
_PROMP = PromptTemplate.from_template("""
    Eres un experto en biología espacial. Tu tarea es responder la pregunta del usuario.
    
    INSTRUCCIONES CLAVE:
    1. Utiliza **exclusivamente** la información de los fragmentos proporcionados en el [CONTEXTO RECUPERADO].
    2. Genera la respuesta **SOLO** en formato JSON que se ajuste estrictamente al esquema JSON proporcionado.
    3. Para la sección 'grafo', extrae los conceptos clave de los documentos recuperados y utiliza su 'metadata' para llenar los campos 'titulo' y 'link'.
    
    Pregunta: {question}
    
    [CONTEXTO RECUPERADO]:
    {context}

    [ESQUEMA JSON REQUERIDO]:
    {json_schema}
""")


# =======================================================
# CLASE DEL AGENTE SEMÁNTICO
# =======================================================
class BiologySemanticAgent:
    def __init__(self, vector_store: VectorStore, llm):
        self.vector_store = vector_store
        self.llm = llm
        
        # Inicializa el parser de JSON
        self.json_parser = JsonOutputParser()
        self.prompt_chain = _PROMP
        
        # Cadena de Generación: Prompt -> LLM (forzado JSON) -> Parser (a dict de Python)
        self.generation_chain = (
            self.prompt_chain 
            | self.llm.bind(response_schema=_JSON_SCHEMA)
            | self.json_parser
        )

    # Método para formatear los documentos (soluciona el AttributeError)
    def _format_context(self, documents: List[Document]) -> str:
        """Convierte una lista de objetos Document en una cadena legible para el LLM, incluyendo metadatos."""
        formatted_context = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Fuente Desconocida')
            name = doc.metadata.get('name', 'Documento sin nombre')

            formatted_context.append(
                f"### Fragmento #{i+1}\n"
                f"**Nombre:** {name}\n"
                f"**Fuente/Link:** {source}\n"
                f"**Contenido:**\n"
                f"{doc.page_content}\n"
            )
        return "\n---\n".join(formatted_context)
    
    
    def generate(self, question: str, k: int) -> Dict[str, Any] | None:
        """
        Genera la respuesta y el grafo en formato JSON.
        Devuelve un diccionario de Python o None si el parseo falla.
        """
        similarity_results = self.vector_store.similarity_search(question, k)
        context_string = self._format_context(similarity_results) 
        
        # Los datos de entrada que necesita el prompt
        input_data = {
            'question': question,
            'context': context_string,
            # Las instrucciones de formato del parser, esenciales para guiar al LLM
            'json_schema': self.json_parser.get_format_instructions() 
        }

        # Ejecutamos la cadena completa
        try:
            # La cadena devuelve el diccionario de Python ya parseado
            return self.generation_chain.invoke(input_data)
        except Exception as e:
            # Captura errores si el LLM devuelve un JSON malformado
            print(f"ERROR FATAL DE PARSEO JSON. El LLM no cumplió el esquema: {e}")
            return None # Devuelve None para evitar un fallo en la función de impresión

'''
class BiologySemanticAgent:
    def __init__(self, vector_store: VectorStore, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.json_parser = JsonOutputParser(pydantic_object=None) # Usamos el parser genérico
        
        # 1. Creamos la cadena (Chain) para la generación JSON
        # Este es el nuevo enfoque clave de LangChain:
        self.chain = (
            self.llm
            .bind(response_schema=_JSON_SCHEMA) # Forzamos el esquema JSON
            | self.json_parser
        )

    def _format_context(self, documents: list) -> str:
        # Usamos la función de formato que creamos antes
        formatted_context = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Fuente Desconocida')
            name = doc.metadata.get('name', 'Documento sin nombre')

            formatted_context.append(
                f"### Fragmento #{i+1}\n"
                f"**Nombre:** {name}\n"
                f"**Fuente/Link:** {source}\n"
                f"**Contenido:**\n"
                f"{doc.page_content}\n"
            )
        return "\n---\n".join(formatted_context)

    # El método generate_stream ya no es útil para JSON estricto, usamos generate
    def generate(self, question: str, k: int):
        similarity_results = self.vector_store.similarity_search(question, k)
        context_string = self._format_context(similarity_results) 
        
        prompt_with_context = _PROMP.invoke(
            {
                'question': question,
                'context': context_string,
                'json_schema': _JSON_SCHEMA # Pasamos el esquema JSON al prompt para que el LLM lo vea
            }
        )
        
        # 2. Invocamos la cadena (LLM + Parser)
        # Esto devolverá un objeto JSON/Diccionario de Python
        return self.llm.invoke(prompt_with_context, 
                               config={"response_schema": _JSON_SCHEMA})

    # Si aún quieres el método de streaming (aunque inestable para JSON):
    def generate_stream(self, question: str, k: int):
        # Para mantener el JSON estricto, es mejor no usar streaming aquí, 
        # o tendrías que manejar el JSON incompleto en el frontend.
        # Devuelve un solo bloque JSON con 'generate'.
        return self.generate(question, k)
-------------------------------------------------------------------------------------
class BiologySemanticAgent:
    def __init__(self, vector_store: VectorStore, llm):
        self.vector_store = vector_store
        self.llm = llm

    def generate(self, question: str, k: int):
        similarity_results = self.vector_store.similarity_search(question, k)
        prompt_with_context = _PROMP.invoke(
            {
                'question': question,
                'context': similarity_results
            }
        )

        return self.llm.invoke(prompt_with_context)
    
    def generate_stream(self, question: str, k: int):
        similarity_results = self.vector_store.similarity_search(question, k)
        prompt_with_context = _PROMP.invoke(
            {
                'question': question,
                'context': similarity_results
            }
        )

        return self.llm.stream(prompt_with_context)
        '''