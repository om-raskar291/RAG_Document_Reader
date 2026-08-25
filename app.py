import os
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Harry Potter RAG",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📚 Harry Potter RAG Document Reader")

st.write(
    "Ask questions based only on the information available "
    "in the Harry Potter document."
)

st.divider()


# ============================================================
# GOOGLE GEMINI API KEY
# ============================================================

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ============================================================
# LOAD AND SPLIT DOCUMENT
# ============================================================

@st.cache_resource
def create_rag_chain():

    file_path = "HarryPotterRag.txt"

    # --------------------------------------------------------
    # Document Loader
    # --------------------------------------------------------

    loader = TextLoader(
        file_path,
        encoding="utf-8"
    )

    docs = loader.load()

    # --------------------------------------------------------
    # Text Splitting
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = splitter.split_documents(docs)


    # --------------------------------------------------------
    # Gemini Embeddings
    # --------------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )


    # --------------------------------------------------------
    # ChromaDB Vector Store
    # --------------------------------------------------------

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="harry_potter_rag"
    )


    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever()


    # --------------------------------------------------------
    # Gemini LLM
    # --------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.5
    )


    # --------------------------------------------------------
    # Prompt Template
    # --------------------------------------------------------

    template = """
Answer the question based only on the following context:

{context}

Question:
{question}

If the answer is not present in the provided context,
say that the information is not available in the document.
"""

    prompt = PromptTemplate.from_template(
        template
    )


    # --------------------------------------------------------
    # Document Formatter
    # --------------------------------------------------------

    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )


    # --------------------------------------------------------
    # LCEL RAG Chain
    # --------------------------------------------------------

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================
# INITIALIZE RAG
# ============================================================

try:

    rag_chain = create_rag_chain()

except Exception as e:

    st.error(
        "Unable to initialize the RAG application."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about Harry Potter..."
)


# ============================================================
# GENERATE ANSWER
# ============================================================

if question:

    # User message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Assistant response

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the document..."
        ):

            try:

                answer = rag_chain.invoke(
                    question
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    "An error occurred while generating the answer."
                )

                st.exception(e)