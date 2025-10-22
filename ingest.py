import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the path for the persistent vector database
PERSIST_DIRECTORY = "db_chroma"

# Define the path to the reference PDF document
SOURCE_DOCUMENT = os.path.join("reference_docs", "nephrology_reference.pdf")

# Load the document
print(f"Loading document: {SOURCE_DOCUMENT}")
loader = PyPDFLoader(SOURCE_DOCUMENT)
documents = loader.load()

# Split the document into chunks
print("Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize the free, local embedding model
print("Initializing embedding model...")
model_name = "all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

# Create the vector database from the text chunks
print("Creating vector database...")
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

# Persist the database to disk
vectordb.persist()

print(f"Ingestion complete! Database saved to {PERSIST_DIRECTORY}")