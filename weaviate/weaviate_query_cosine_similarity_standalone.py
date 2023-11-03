import json
import os
import requests
from dotenv import load_dotenv

# Load the .env file where the environment variables are set
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Load environment variables from .env
WEAVIATE_URL = os.getenv('2023oct25_WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv("2023oct21_WEAVIATE_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("2023oct21_AZURE_OPENAI_API_KEY")

# Set end-point url and headers
url = f"{WEAVIATE_URL}/v1/graphql"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {WEAVIATE_API_KEY}', 
    'X-Azure-Api-Key': f'{AZURE_OPENAI_API_KEY}',
}

# Define global constants
GLOBAL_CLASS_NAME = 'Hobbit'
ENDING_QUERY = 'who is thorin'
SIMILARITY_THRESHOLD = 0.7
PROPERTY_NAME = 'data_chunk'
TOP_N = 1

def query_weaviate():
    # Construct the GraphQL payload
    graphql_query = {
        "query": f'''
        {{
            Get {{
                {GLOBAL_CLASS_NAME}(nearText: {{
                    concepts: ["{ENDING_QUERY}"],
                    certainty: {SIMILARITY_THRESHOLD}
                }}, limit: {TOP_N}) {{
                    {PROPERTY_NAME}
                    _additional {{certainty}}
                }}
            }}
        }}
        '''
    }
    # Send the request and get the response
    response = requests.post(url, headers=headers, json=graphql_query)

    # Pretty print the JSON response
    print("Response from Weaviate query:")
    print(json.dumps(response.json(), indent=4))
    print()

if __name__ == "__main__":
    query_weaviate()