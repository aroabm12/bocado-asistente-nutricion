# ---------- IMPORTACIONES ----------
import os
import re
import uuid
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# configuro la página
st.set_page_config(page_title="Bocado", page_icon="🥗")
st.title("🥗 Bocado")
st.caption("Asistente experto en nutrición (RAG con Gemini, ChromaDB y LangGraph)")

# ---------- CLAVE DE API ----------
# en local la leo del .env; en Streamlit Cloud, de los "Secrets"
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        API_KEY = None

# si no hay clave, aviso y paro
if not API_KEY:
    st.error("Falta la clave GEMINI_API_KEY (archivo .env o Secrets de Streamlit).")
    st.stop()

# ---------- DOCUMENTOS ----------
DOCS = {
    "data/Nutricion_1_Fundamentos.pdf": "Fundamentos de la Nutrición",
    "data/Nutricion_2_Alimentacion_Saludable.pdf": "Alimentación Saludable y Dieta Mediterránea",
    "data/Nutricion_3_Habitos_y_Mitos.pdf": "Hábitos, Menús y Mitos",
}

# ---------- PROMPT DEL SISTEMA ----------
SYSTEM_PROMPT = """Eres Bocado, un asistente experto en nutrición y alimentación saludable.

OBJETIVO
Explicar de forma clara y sencilla conceptos de nutrición usando SOLO el contexto que te doy.

REGLAS
1. Responde únicamente con la información de <contexto>. No inventes datos ni cifras.
2. Si el contexto no tiene la respuesta, di exactamente: "No tengo información suficiente en mis documentos para responder a eso."
3. Si la pregunta no trata de nutrición o alimentación, di que solo puedes ayudar con esos temas.
4. Empieza directamente con la respuesta. No saludes.
5. Usa un lenguaje sencillo y respuestas cortas (máximo 6 frases). Usa listas solo si ayudan.
6. No des diagnósticos, ni dietas personalizadas, ni dosis de suplementos. Si la persona tiene una enfermedad, está embarazada o pide algo médico, recomienda consultar a un médico o dietista-nutricionista.
7. Si has respondido con el contexto, termina con una línea: "Fuentes: [Documento, pág. X]". Si no has usado el contexto (por ejemplo, si rechazas la pregunta), no pongas fuentes.

<contexto>
{contexto}
</contexto>"""


class Estado(TypedDict):
    messages: Annotated[list, add_messages]
    pregunta_autonoma: str
    contexto: str
    fuentes: list


def extraer_texto(contenido):
    # convierto la respuesta de Gemini en texto normal
    if isinstance(contenido, str):
        return contenido
    partes = []
    for bloque in contenido:
        if isinstance(bloque, dict) and "text" in bloque:
            partes.append(bloque["text"])
        elif isinstance(bloque, str):
            partes.append(bloque)
    return "".join(partes)


def limpiar(texto):
    # limpio el texto de cada página
    texto = re.sub(r"-\n", "", texto)
    texto = re.sub(r"^\s*\d+\s*$", "", texto, flags=re.M)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# creo el agente UNA sola vez (st.cache_resource evita repetirlo en cada mensaje)
@st.cache_resource(show_spinner="Preparando la base de conocimiento...")
def crear_agente(api_key):
    # 1. leo los PDFs
    paginas = []
    for fichero, nombre in DOCS.items():
        for i, pagina in enumerate(PyPDFLoader(fichero).load()):
            pagina.metadata["documento"] = nombre
            pagina.metadata["pagina"] = i + 1
            pagina.page_content = limpiar(pagina.page_content)
            if len(pagina.page_content) >= 80:
                paginas.append(pagina)

    # 2. troceo
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(paginas)

    # 3. guardo en Chroma (en memoria, se rehace al arrancar)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=api_key)
    vectorstore = Chroma(collection_name="nutricion_web", embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key)

    # paso 1: reformular
    def nodo_reformular(estado: Estado):
        mensajes = estado["messages"]
        pregunta = mensajes[-1].content
        if len(mensajes) == 1:
            return {"pregunta_autonoma": pregunta}
        historial = "\n".join(
            f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {extraer_texto(m.content)}"
            for m in mensajes[:-1]
        )
        instruccion = (
            "Reescribe la última pregunta del usuario para que se entienda sin la conversación. "
            "Devuelve SOLO la pregunta reescrita, sin explicaciones.\n\n"
            f"Conversación:\n{historial}\n\nÚltima pregunta: {pregunta}"
        )
        respuesta = llm.invoke(instruccion)
        return {"pregunta_autonoma": extraer_texto(respuesta.content).strip()}

    # paso 2: recuperar
    def nodo_recuperar(estado: Estado):
        docs = retriever.invoke(estado["pregunta_autonoma"])
        contexto = "\n\n".join(
            f"[{d.metadata['documento']}, pág. {d.metadata['pagina']}]\n{d.page_content}"
            for d in docs
        )
        fuentes = sorted({f"{d.metadata['documento']}, pág. {d.metadata['pagina']}" for d in docs})
        return {"contexto": contexto, "fuentes": fuentes}

    # paso 3: generar
    def nodo_generar(estado: Estado):
        sistema = SystemMessage(content=SYSTEM_PROMPT.format(contexto=estado["contexto"]))
        respuesta = llm.invoke([sistema] + estado["messages"])
        return {"messages": [AIMessage(content=extraer_texto(respuesta.content))]}

    grafo = StateGraph(Estado)
    grafo.add_node("reformular", nodo_reformular)
    grafo.add_node("recuperar", nodo_recuperar)
    grafo.add_node("generar", nodo_generar)
    grafo.add_edge(START, "reformular")
    grafo.add_edge("reformular", "recuperar")
    grafo.add_edge("recuperar", "generar")
    grafo.add_edge("generar", END)
    return grafo.compile(checkpointer=InMemorySaver())


agente = crear_agente(API_KEY)

# ---------- CHAT ----------
# cada visitante tiene su propia conversación (su propio thread_id)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "historial" not in st.session_state:
    st.session_state.historial = []

# barra lateral
with st.sidebar:
    st.subheader("Sobre Bocado")
    st.write("Responde solo con 3 documentos sobre nutrición y recuerda la conversación.")
    st.write("No sustituye a un médico ni a un dietista-nutricionista.")
    if st.button("Nueva conversación"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.historial = []
        st.rerun()

# muestro los mensajes anteriores
for rol, texto in st.session_state.historial:
    with st.chat_message(rol):
        st.markdown(texto)

# caja para escribir
pregunta = st.chat_input("Pregunta sobre nutrición...")
if pregunta:
    st.session_state.historial.append(("user", pregunta))
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            resultado = agente.invoke({"messages": [HumanMessage(content=pregunta)]}, config=config)
            respuesta = resultado["messages"][-1].content
        st.markdown(respuesta)
    st.session_state.historial.append(("assistant", respuesta))