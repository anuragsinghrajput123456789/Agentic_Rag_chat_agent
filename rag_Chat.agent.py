from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

load_dotenv()

# session state
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# processing function
def process_document(path):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = splitter.split_documents(documents=docs)

    # FIXED embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

    # FIXED groq model
    llm = ChatGroq(model_name="llama3-70b-8192")

    # FIXED tool name consistency
    @tool
    def retriver_context(query: str):
        """
        Retrive documents relevant to a query from the knowledge base
        """
        context = ""
        docs = vector_db.similarity_search(query=query, k=3)

        for doc in docs:
            context += doc.page_content + "\n\n"

        return context

    # FIXED prompt tool name
    system_prompt = """You are a helpful assistant that answers questions using retrival context
My knowledge based consists of the details from the uploaded documents.
Always use the `retriver_context` tool for questions requiring external knowledge."""

    # FIXED: removed checkpointer (was causing crash)
    agent = create_agent(
        model=llm,
        tools=[retriver_context],
        system_prompt=system_prompt,
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# upload UI
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(
        label="Select PDF Files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded:
        with st.spinner("Processing..."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()


# chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    query = st.chat_input("Ask anything related to uploaded documents...")

    if query:
        st.chat_message("user").markdown(query)

        # FIXED invoke (no thread config needed now)
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
