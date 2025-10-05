import streamlit as st

# Importaciones para Gemini
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

# Asegúrate de que esta importación sea correcta y que la clase esté disponible
from agentes.Agente_semantico import BiologySemanticAgent 


HUMAN = "HUMAN"
AI = "AI"

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(
    page_title="Agente Experto en Biología Espacial", 
    page_icon="🔬"
)


class StreamlitUI:
    def __init__(self):
        # Carga las variables de entorno (incluyendo GEMINI_API_KEY)
        load_dotenv('.env') 
        self.__init_semantic_agent()

    def __init_semantic_agent(self):
        if "semantic_agent" not in st.session_state:
            
            # --- 1. Inicializar Gemini LLM ---
            gemini_llm = ChatGoogleGenerativeAI(
                model='gemini-2.5-flash',
                # LangChain busca la API Key en el entorno
            )
            
            # --- 2. Inicializar Embeddings (Ollama/mxbai) ---
            embeddings_generator = OllamaEmbeddings(
                model='mxbai-embed-large'
            )

            # --- 3. Inicializar Vector Store (Chroma) ---
            vector_store = Chroma(
                # Cambiar el nombre de la colección a 'biology'
                collection_name='biology', 
                embedding_function=embeddings_generator,
                persist_directory='./database/chromadb'
            )

            # --- 4. Inicializar el Agente de Biología ---
            # Ahora usa la clase y el LLM de Biología
            agent = BiologySemanticAgent( 
                vector_store,
                gemini_llm
            )

            st.session_state.semantic_agent = agent
    
    # --- Adaptación de la Interfaz ---

    def display_sidebar(self):
        # Revisa que la ruta de la imagen sea correcta en tu proyecto
        #st.sidebar.image('./assets/siafi-logo-blanco.webp', use_container_width=True) 

        st.sidebar.title("🔬 Agente Experto en Biología Espacial")

        st.sidebar.markdown(
            """
            Este proyuecto nace como respuesta a la necesidad de resumir informacion sobre documentacion sobre exploracion lunar para poder acceder de manera mas sencilla y coon una herramienta amigable
            """
        )

        st.sidebar.markdown("SpaceApps Challenge")

    def display_chat_history(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message['type']):
                st.markdown(message["content"])


    def handle_human_message(self, message: str):
        st.chat_message(HUMAN).markdown(message)

        st.session_state.messages.append(
            {
                "type": HUMAN,
                "content": message
            }
        )

    def handle_ai_message(self, prompt: str):
        agent = st.session_state.semantic_agent
        # Se sigue pasando '5' como el número de chunks a recuperar (k)
        response = agent.generate_stream(prompt, 5) 

        with st.chat_message(AI):
            # st.write_stream maneja la salida en tiempo real
            chat_response = st.write_stream(response) 

        st.session_state.messages.append(
            {
                "type": AI,
                "content": chat_response
            }
        )

    def run(self):
        st.title("Agente de Biología Espacial 🪐")
        self.display_sidebar()

        self.display_chat_history()

        prompt = st.chat_input("Mensaje al Agente Biológico...")
        if prompt:
            self.handle_human_message(prompt)
            self.handle_ai_message(prompt)


if __name__ == "__main__":
    ui = StreamlitUI()
    ui.run()