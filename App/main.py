import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader, CSVLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from fpdf import FPDF
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from prompt import prompt_template_text
# import sys
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


class RAGAssistant:
    def __init__(self):
        # self.load_env_variables()
        self.setup_prompt_template()
        self.retriever = None
        self.relative_path = 'data'
        self.filename = 'a.csv'
        self.absolute_path = os.path.join(self.relative_path, self.filename)
        # self.absolute_path = os.path.join(self.relative_path, self.filename)
        self.initialize_retriever(self.absolute_path)
        # self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.llm = ChatOllama(model="Mistral", temperature=0.7)

    def setup_prompt_template(self):
        """Sets up the prompt template for chat completions."""
        self.template = prompt_template_text
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=self.template,
        )

    def initialize_retriever(self, directory_path):
        """Initializes the retriever with documents from the specified directory path."""
        loader = CSVLoader(directory_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        vectbd = Chroma.from_documents(
        documents=docs,
        collection_name="rag-chroma",
        embedding=embeddings,
        )
        self.retriever = vectbd.as_retriever()

    def finetune(self, file_path):
        """Determines the document type and uses the appropriate loader to fine-tune the model."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type.")

        documents = loader.load_and_split() if hasattr(
            loader, 'load_and_split') else loader.load()

        self.process_documents(documents)

    def process_documents(self, documents):
        """Process and index the documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectbd = Chroma.from_documents(
        documents=docs,
        collection_name="rag-chroma",
        embedding=embeddings,
        )
        self.retriever = vectbd.as_retriever()

    def chat(self, user_input):
        """Starts a chat session with the AI assistant."""
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={"verbose": False, "prompt": self.prompt_template,
                               "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
        )

        assistant_response = chain.invoke(user_input)
        response_text = assistant_response['result']
        return response_text


def main():
    assistant = RAGAssistant()

    st.set_page_config(page_title="AI Chat Assistant", layout="wide")
    st.title("AI Chat Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "toc_content" not in st.session_state:
        st.session_state.toc_content = ""
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    option = st.sidebar.selectbox(
        "Choose an option", ("Chat", "Fine-tuning"))

    if option == "Chat":
        st.header("Chat with your Docs")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter your message:"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = assistant.chat(prompt)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)

    elif option == "Fine-tuning":
        st.header("Upload your data here")
        uploaded_file = st.file_uploader(
            "Upload a file for fine-tuning", type=["txt", "pdf", "csv", "xlsx", "docx"])

        if uploaded_file is not None:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Fine-tuning in progress..."):
                assistant.finetune(file_path)
            st.success(
                "Fine-tuning done successfully. You can now chat with the updated RAG Assistant.")


if __name__ == "__main__":
    main()