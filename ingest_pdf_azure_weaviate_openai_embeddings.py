import os
import nltk
import json
import re
import requests
import datetime
import pandas as pd
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

# Global variables
MAX_CHUNK_LEN = 512
PDF_FILE_PATH = "the-hobbit.pdf"
global_class_name = "hobbit"

class WeaviateInterface:
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv('2023oct21_WEAVIATE_URL')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("2023oct21_WEAVIATE_API_KEY")}',
            'X-Azure-Api-Key': f'{os.getenv("2023oct21_AZURE_OPENAI_API_KEY")}',
        }
        self.class_name = global_class_name
        self.deployment_id_gpt = os.getenv("2023oct22_TERADATA_ESC_CLOUD_POC_GPT")
        self.deployment_id_embedding = os.getenv("2023oct22_TERADATA_ESC_CLOUD_POC_EMBEDDING")
        self.resource_name = os.getenv("2023oct22_TERADATA_ESC_CLOUD_POC")

    def create_class(self):
        tillo_test_class = {
            "class": self.class_name,
            "description": "A class to test Weaviate with Vectorization and OpenAI",
            "properties": [
                {
                    "name": "prompt",
                    "description": "A prompt to be added to Weaviate",
                    "dataType": ["string"]
                }],
            "moduleConfig": {
                "generative-openai": {
                    "deploymentId": self.deployment_id_gpt,
                    "model": "gpt-3.5-turbo",
                    "resourceName": self.resource_name},
                "text2vec-openai": {
                    "baseURL": "https://api.openai.com",
                    "deploymentId": self.deployment_id_embedding,
                    "model": "ada",
                    "modelVersion": "002",
                    "resourceName": self.resource_name,
                    "type": "text",
                    "vectorizeClassName": True}}}
        url = self.base_url + '/v1/schema'
        response = requests.post(url, headers=self.headers, data=json.dumps(tillo_test_class))
        print(f"Response code for {self.class_name} class creation:", response.status_code)
        print("Response body:", response.json())
    
    def does_class_exist(self):
        url = self.base_url + '/v1/schema'
        response = requests.get(url, headers=self.headers)
        schema = response.json()
        return self.class_name in [classe['class'] for classe in schema['classes']]

    def create_object(self, prompt):
        data = {
            "class": self.class_name,
            "properties": {
                "prompt": prompt}}
        url = self.base_url + '/v1/objects'
        response = requests.post(url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            objectId = response.json()['id']
            print(f"Sentence '{prompt}' \n...has been successfully added to Weaviate with ID {objectId}.")
        else:
            print(f"There was a problem adding the sentence '{prompt}' to Weaviate. Response code: {response.status_code}")

    def get_existing_prompts(self):
        url = self.base_url + '/v1/graphql'
        graphql_query = {
            "query": f"""
                {{
                    Get {{
                        {self.class_name} {{
                            prompt
                        }}
                    }}
                }}
            """
        }
        response = requests.post(url, headers=self.headers, data=json.dumps(graphql_query))
        existing_objects = response.json()
        try:
            return [obj["prompt"] for obj in existing_objects["data"]["Get"][self.class_name]]
        except KeyError:
            print(f"No existing data found for class: {self.class_name}")
            return []

def normalize_text(s):
    s = s.replace('\n', ' ')  
    s = re.sub(r'[^A-Za-z0-9. ]+', '', s)  
    s = re.sub(' +', ' ', s)  
    return s

def split_into_sentences(text, max_chunk_len=MAX_CHUNK_LEN):
    print(f"\nSplitting text into sentences of size {MAX_CHUNK_LEN}, this may take a few moments...")
    sentences = nltk.sent_tokenize(text)
    return sentences

def split_into_chunks(text, max_chunk_len=MAX_CHUNK_LEN):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence.split(' ')) <= max_chunk_len:
            chunk += " " + sentence
        else:
            chunks.append(chunk)
            chunk = sentence

    chunks.append(chunk)
    return chunks

def process_text(file_path=PDF_FILE_PATH):
    text = normalize_text(extract_text(file_path))
    chunks = split_into_chunks(text)
    return chunks

def main():
    interface = WeaviateInterface()

    if not interface.does_class_exist():
        interface.create_class()
    else:
        print(f"{interface.class_name} class already exists, skipping class creation.")
    
    existing_prompts = interface.get_existing_prompts()
    chunks = process_text()

    for i, chunk in enumerate(chunks):
        if chunk not in existing_prompts:
            interface.create_object(chunk)
        print(f"Progress: added chunk {i+1} of {len(chunks)}")
    
if __name__ == "__main__":
    main()