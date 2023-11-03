import os
import nltk
import json
import re
import requests
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import time

# Global variables
# The user's ending query or the question that the model is supposed to answer.
ENDING_QUERY = "Who is Thorin"

# The number of chunks that will be processed and stored in Weaviate. 
# If set to None or 0, all chunks will be processed.
ONLY_PROCESS_CHUNK_NUMBER = 0

# The number of top results that the model should return after performing a search.
TOP_N = 6

# The maximum length of the text chunks that will be processed. The chunks will be tokenized sentences.
MAX_CHUNK_LEN = 512

# The path to the PDF file that will be processed and stored in Weaviate.
PDF_FILE_PATH = "../books/the-hobbit.pdf"

# The name of the class in Weaviate where data chunks will be stored.
GLOBAL_CLASS_NAME = "Hobbit"

# The minimum similarity threshold to consider when the model is looking for similar chunks in Weaviate.
SIMILARITY_THRESHOLD = 0.7

# A name chosen for the property in the 'Hobbit' class in Weaviate. This property will store discrete chunks of the PDF document.
PROPERTY_NAME = "data_chunk"

# A detailed explanation of the purpose and content of the 'data_chunk' property. 
# This description states that 'data_chunk' holds sections of the PDF content that have been processed and added to Weaviate.
PROPERTY_DESCRIPTION = "A section of the processed text from the PDF document that is added to Weaviate for further querying."

# Name of the class created in Weaviate for storing, querying and analysing the data.
CLASS_DESCRIPTION = "Class for storing processed sections of a book text based on user-defined configuration. Queries leveraging vectorization and OpenAI's language models can be performed on the data in this class."

class WeaviateInterface:
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv('2023oct25_WEAVIATE_URL')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("2023oct21_WEAVIATE_API_KEY")}',
            'X-Azure-Api-Key': f'{os.getenv("2023oct21_AZURE_OPENAI_API_KEY")}',
        }
        self.class_name = GLOBAL_CLASS_NAME
        self.deployment_id_gpt = os.getenv("2023oct22_TERADATA_ESC_CLOUD_POC_GPT")
        self.deployment_id_embedding = os.getenv("2023oct22_TERADATA_ESC_CLOUD_POC_EMBEDDING")
        self.resource_name = os.getenv("2023oct22_TERADATA_ESC_CLOUD_POC")

    def create_class(self):
        GLOBAL_CLASS_STRUCTURE = {
            "class": self.class_name,
            "description": CLASS_DESCRIPTION,
            "properties": [
                {
                    "name": PROPERTY_NAME,
                    "description": PROPERTY_DESCRIPTION,
                    "dataType": ["string"]}],
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
        response = requests.post(url, headers=self.headers, data=json.dumps(GLOBAL_CLASS_STRUCTURE))
        print(f"Response code for {self.class_name} class creation:", response.status_code)
        print("Response body:", response.json())
    
    def does_class_exist(self):
        url = self.base_url + '/v1/schema'
        response = requests.get(url, headers=self.headers)
        try:
            schema = response.json() 
            return self.class_name in [classe['class'] for classe in schema['classes']]
        except json.decoder.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            return False
    
    def process_chunks(self, chunks, existing_prompts):
        if ONLY_PROCESS_CHUNK_NUMBER == 0 or ONLY_PROCESS_CHUNK_NUMBER is None:
            total_chunks = len(chunks)
        else:
            total_chunks = min(len(chunks), ONLY_PROCESS_CHUNK_NUMBER)
            
        for i, chunk in enumerate(chunks[:total_chunks]):
            if chunk not in existing_prompts:
                self.create_object(chunk)
                print(f"Progress: Added chunk {i+1} of {total_chunks} to Weaviate.")
            else:
                print(f"Progress: Skipped duplicated chunk {i+1} of {total_chunks}.")
        return total_chunks

    def create_object(self, prompt):
        data = {
            "class": self.class_name,
            "properties": {
                PROPERTY_NAME: prompt}}
        url = self.base_url + '/v1/objects'
        response = requests.post(url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            objectId = response.json()['id']
            print(f"Chunk '{prompt[:50]}...' has been successfully added to Weaviate with ID {objectId}")
        else:
            print(f"There was a problem adding the chunk '{prompt[:50]}...' to Weaviate. Response code: {response.status_code}")

    def query_weaviate(self, query_text=ENDING_QUERY, top_n=TOP_N):
        url = self.base_url + '/v1/graphql'
        graphql_query = {
            "query": f"""
            {{
                Get {{
                    {self.class_name}(
                        nearText: {{
                            concepts: ["{query_text}"],
                            certainty: {SIMILARITY_THRESHOLD}
                        }},
                        limit: {top_n}) {{
                            {PROPERTY_NAME}
                            _additional {{certainty}}
                    }}
                }}
            }}
            """
        }
        response = requests.post(url, headers=self.headers, json=graphql_query)
        search_response = response.json()
        certainty_scores = [
            item["_additional"]["certainty"] 
            for item in search_response["data"]["Get"][self.class_name]
        ]
        top_scoring_text = search_response["data"]["Get"][self.class_name][0][PROPERTY_NAME]
        return len(certainty_scores), max(certainty_scores), min(certainty_scores), top_scoring_text

    def get_existing_prompts(self):
        url = self.base_url + '/v1/graphql'
        graphql_query = {
            "query": f"""
                {{
                    Get {{
                        {self.class_name} {{
                            {PROPERTY_NAME}
                        }}
                    }}
                }}
            """
        }
        response = requests.post(url, headers=self.headers, data=json.dumps(graphql_query))
        existing_objects = response.json()
        try:
            return [obj[PROPERTY_NAME] for obj in existing_objects["data"]["Get"][self.class_name]]
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
    print(f"Extracted {len(chunks)} chunks from text.")
    return chunks

def process_text(file_path=PDF_FILE_PATH):
    text = normalize_text(extract_text(file_path))
    chunks = split_into_chunks(text)
    return chunks

def main():
    start_time = time.time()
    
    interface = WeaviateInterface()
    if not interface.does_class_exist():
        print(f"{interface.class_name} class does not exist. Creating class...")
        interface.create_class()
    else:
        print(f"The '{interface.class_name}' class already exists. Skipping class creation...")
    
    print("\nStep 1: Checking for existing data chunks in Weaviate...")
    existing_prompts = interface.get_existing_prompts()
    if existing_prompts:
        print(f"Found {len(existing_prompts)} existing data chunks in the '{interface.class_name}' class in Weaviate.")
    else:
        print("No existing data chunks found.")

    print("\nStep 2: Extracting and processing text from the PDF document...")
    chunks = process_text()
    print(f"Successfully extracted and processed {len(chunks)} text chunks from the document.")

    print(f"Adding chunks of text to Weaviate. Each chunk is added as an individual object in the '{interface.class_name}' class...")
    total_processed = interface.process_chunks(chunks, existing_prompts)
    print(f"Done processing. Total chunks processed and added to Weaviate: {total_processed}")

    print("\nStep 4: Querying Weaviate for data that is similar to the user's query...\n")
    print(f"Response and score from Weaviate on best match: '{interface.query_weaviate(ENDING_QUERY)}'")
    num_responses, top_score, bottom_score, top_scoring_text = interface.query_weaviate(ENDING_QUERY)

    end_time = time.time()
    run_time = end_time - start_time
    print("")
    print("- - - SUMMARY - - -")
    print("")
    print("1. USER QUERY:")
    print(f"The phrase or question you asked Weaviate to analyze was: '{ENDING_QUERY}'")
    print("")
    print("2. SEARCH RESULTS:")
    print(f"Number of results from Weaviate found that were similar to your query: {num_responses}")
    print("")
    print("3. MATCH QUALITY:")
    print(f"The best matching passage from Weaviate had a cosine similarity score of {top_score}.")
    print(f"The lowest cosine similarity score out of your top results was {bottom_score}.")
    print("")
    print("4. BEST MATCH:")
    print(f"The passage that best matched your query was: '{top_scoring_text[:500]}...'")
    print("")
    print("5. RUN TIME:") 
    print(f"The total time it took to process your query and retrieve results: {run_time} seconds.")

if __name__ == "__main__":
    main()