import requests
import json
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

load_dotenv(dotenv_path)

base_url = os.getenv('2023oct25_WEAVIATE_URL')
url = f"{base_url}/v1/graphql"  # append additional part to the base URL

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.getenv("2023oct21_WEAVIATE_API_KEY")}', 
    'X-Azure-Api-Key': f'{os.getenv("2023oct21_AZURE_OPENAI_API_KEY")}',
}

# Define global constants
CLASS_NAME = 'Hobbit'
SEARCH_TERM = 'what is this about'
SIMILARITY_THRESHOLD = 0.7
OBJECT_VALUE = 'prompt'
LIMIT = 1

def search_vector_similarity():
    payload = {
        "query": f'''
        {{
            Get {{
                {CLASS_NAME}(nearText: {{
                    concepts: ["{SEARCH_TERM}"],
                    certainty: {SIMILARITY_THRESHOLD}
                }}, limit: {LIMIT}) {{
                    {OBJECT_VALUE}
                    _additional {{certainty}}
                }}
            }}
        }}
        '''
    }
    response = requests.post(url, headers=headers, json=payload)
    print("Similarity Search:")
    print(json.dumps(response.json(), indent=4))
    print()

if __name__ == "__main__":
    search_vector_similarity()