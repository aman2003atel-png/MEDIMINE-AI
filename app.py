from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import RateLimitError
from dotenv import load_dotenv
from flask_cors import CORS
from src.prompt import *
import os
                                                                                                                                                            
app = Flask(__name__)
CORS(app)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = chatModel = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=300,        
    timeout=30,
    max_retries=0          
)


system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use the provided context to answer the question. "
    "If you do not know the answer, say you do not know. "
    "Use a maximum of three sentences and keep the answer concise."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@retry(
    retry=lambda e: isinstance(e, RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def run_rag(query: str):
    return rag_chain.invoke({"input": query})



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    response = run_rag(msg)
    print("Response : ", response["answer"])
    return str(response["answer"])

 

@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


