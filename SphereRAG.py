import os

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.llms import GPT4All
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

#Let's do some retrieval augmented generation: here's some details on Crisis Management Exercises in a doc.
loader = PyPDFLoader("Docs/Sphere-Handbook-2018-EN.pdf")
data = loader.load()

#We split the document up into manageable slices. Important for large docs. chunk_overlap can be useful to provide better coverage.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Now we make a vectorstore out of the chunked data
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=4)

docs = retriever.invoke("How do we get food to displaced isolated communities?")

print(docs)

if __name__ == "__main__":
    main()
