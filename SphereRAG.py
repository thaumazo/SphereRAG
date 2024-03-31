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

def main():
    # Retrieve the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please check your .env file.")
    
    # Initialize the OpenAI LLM with your API key and specify the model
    llm = OpenAI(api_key=api_key, model_name="gpt-3.5-turbo-instruct")

    #Maybe you want to use a local model...
    #local_path = ("../models/orca-2-7b.Q4_0.gguf")  # replace with your desired local file path)
    #llm = GPT4All(model=local_path)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant in creating all aspects of Crisis Management Exercises. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm
    
    Chat_History = ChatMessageHistory()
    Chat_History.add_user_message("Write a short crisis response scenario with 1 inject suitable for wildfire training for paramedics and firefighters.")
    
    response = chain.invoke({"messages":Chat_History.messages})
    Chat_History.add_ai_message(response);
    print(response) 
    
#    Chat_History.add_user_message("Repeat your previous response in French!")
#    response = chain.invoke({"messages": Chat_History.messages})
#    print(response)
    

if __name__ == "__main__":
    main()
