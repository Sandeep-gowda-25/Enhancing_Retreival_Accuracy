from llm_client import LLMOperations
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
load_dotenv('.env')
import os

def create_vectors(text_chunks):
    llm = LLMOperations()
    vectors = []
    for i,chunk in enumerate(text_chunks):
        vectors.append({
            "id":str(i),
            "values":llm.get_embeddings(chunk),
            "metadata":{'text': chunk}
        })
    return vectors

def create_vectors_processed(original_chunks,processed_chunks):
    llm = LLMOperations()
    vectors = []
    for i,chunk in enumerate(processed_chunks):
        vectors.append({
            "id":str(i),
            "values":llm.get_embeddings(chunk),
            "metadata":{'text': original_chunks[i]}
        })
    return vectors

def create_embeddings(text):
    llm = LLMOperations()
    return llm.get_embeddings(text)

def store_vectors_unaltered(vectors):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "mahabharata"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name,dimension=1536,
            spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ))
    
    index = pc.Index(index_name)
    index.upsert(vectors=vectors,namespace="unaltered")
    return index

def store_vectors_processed(vectors):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "mahabharata"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name,dimension=1536,
            spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ))
    
    index = pc.Index(index_name)
    index.upsert(vectors=vectors,namespace="processed")
    return index

def delete_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "mahabharata"
    pc.delete_index(index_name)