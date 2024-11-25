import openai
from openai import AzureOpenAI, embeddings
from dotenv import load_dotenv
load_dotenv('.env')
import os

class LLMOperations:
    '''
    Thic calls initializes LLM client with required configurations and
    gives methods to get response and also to get embededdings of given text input
    '''

    def __init__(self,*args,**kwargs):
        self.api_key = os.getenv("OPENAI_APIKEY")
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.azure_client = AzureOpenAI(api_key=self.api_key,azure_endpoint=self.openai_endpoint,api_version="2024-05-01-preview")
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        self.deployment_name = os.getenv("DEPLOYMENT_NAME")
        system_message = "You are an AI assistant that provides right information."
        self.messages = [{"role":"system","content":system_message}]

    def get_response(self,query:str):
        self.messages.append({"role":"user","content":query})
        response = self.azure_client.chat.completions.create(model=self.deployment_name,
        messages=self.messages
        )
        out = response.choices[0].message.content
        self.messages.append({"role":"assistant","content":out})
        return out
    
    def get_embeddings(self,input:str):
        embeddings = self.azure_client.embeddings.create(input=input,model=self.embedding_model).data[0].embedding
        return embeddings

