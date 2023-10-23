import datetime
import nltk
import os
import pandas as pd
import re
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from openai.embeddings_utils import get_embedding, cosine_similarity
from pdfminer.high_level import extract_text
from time import sleep
import openai

#start the timer
start_time = datetime.datetime.now()

load_dotenv()
nltk.download('punkt')

# Global variables
# Mostly for test, this controls how many chunks of the text to process. If set to None, all the chunks from the text will be processed. 
# If set to a numeric value (for example, 6), only that number of initial chunks will be processed.
ONLY_PROCESS_CHUNK_NUMBER = 500

# TOKEN_COST is the cost per token for using OpenAI's API service. As of October 2023, each token costs $0.0001.
TOKEN_COST = 0.0001

# Maximum length for a chunk of text. 
# Consequently, the maximum length of a chunk affects the structure and granularity of our document processing and embedding calculation.
MAX_CHUNK_LEN = 512

# TOP_N represents the number of top relevant sentences/answers we want to retrieve from the document for the given user query.
# For instance, if TOP_N is 6, we will get the top 6 relevant sentences from the document.
TOP_N = 6

# This is the path of the PDF file to process. The content from this file will be extracted, split into chunks, and processed to get the embeddings.
PDF_FILE_PATH = "the-hobbit.pdf"

#This is the question or query you are trying to answer from the ingested PDF.
USER_QUERY = "Did Bilbo Baggins ever regret his decision to join the dwarves and Gandalf on their epic adventure to capture the dragon's hoard?"

# This determines the number of retry attempts the program will make when retrieving embeddings if there's a failure.
RETRIES = 5

# token_count is a counter to keep track of the number of tokens processed.
token_count = 0

print("\n------------------START------------------")
print(f"Start Time: {str(start_time)}")
print(f"Processing text file: {PDF_FILE_PATH}")
print(f"Processing text file: {PDF_FILE_PATH}")


ADA_V2 = os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT")
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_EMBEDDINGS_API_KEY")
openai.api_base = os.getenv("OPENAI_EMBEDDINGS_ENDPOINT")
openai.api_version = os.getenv("OPENAI_API_VERSION")

def normalize_text(s):
    s = s.replace('\n', ' ')  # Replace line breaks with a space
    s = re.sub(r'[^A-Za-z0-9. ]+', '', s)  # Remove special characters excluding periods
    s = re.sub(' +', ' ', s)  # Replace multiple spaces with a single space
    return s

def get_embedding_with_retries(chunk, model, retries=RETRIES, chunk_num=0, total_chunks=0):
    global token_count
    for _ in range(retries):
        try:
            print("--------------------------------------------------------------------------------")
            print(f"Processing chunk {chunk_num+1} of {total_chunks}...\nVectorizing this text: ", chunk[:50000]+"...")
            #// show full chunk print("Processing chunk {chunk_num+1} of {total_chunks}...\nVectorizing this text: ", chunk)

            resp = get_embedding(chunk, model)
            print("First 3 vectors of the embedding:", str(resp[:3])+"...")
            token_count += len(chunk.split())
            return resp
        except Exception as e:
            print(f"\n*** Exception occurred while getting embedding for chunk {chunk_num+1}. Error detail: {str(e)}. Retrying... ***")
            sleep(1)
    raise Exception(f"\n** Failed to get embedding for chunk {chunk_num+1} after all retries. **")

def split_into_chunks(text, max_chunk_len=MAX_CHUNK_LEN):
    print(f"\n------------------SPLITTING TEXT INTO CHUNKS START------------------")
    print(f"Splitting text into chunks of size {MAX_CHUNK_LEN}\nThis may take a few moments...\n")
    
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

    print(f"\n------------------SPLITTING TEXT INTO CHUNKS END------------------")
    print(f"Completed splitting text into chunks. Total chunks created: {len(chunks)}")
    
    return chunks

def process_text(file_path=PDF_FILE_PATH):
    text = normalize_text(extract_text(file_path))
    chunks = split_into_chunks(text)
    if ONLY_PROCESS_CHUNK_NUMBER is not None:
        chunks = chunks[:ONLY_PROCESS_CHUNK_NUMBER]
    return chunks

def compute_embeddings(df, model=ADA_V2):
    print("\n------------------COMPUTE EMBEDDINGS TRANSFORMATION START------------------")
    print(f"Transforming text chunks into embeddings...")
    
    total_chunks = min(len(df), ONLY_PROCESS_CHUNK_NUMBER if ONLY_PROCESS_CHUNK_NUMBER is not None else len(df))
    df['embeddings'] = [None]*len(df)
    for i, row in enumerate(df.itertuples()):
        if i >= total_chunks:
            break
        print("--------------------------------------------------------------------------------")
        print(f"Getting embeddings for chunk {i+1} of {total_chunks}")
        df.at[row.Index, 'embeddings'] = get_embedding_with_retries(row.text, model, chunk_num=i, total_chunks=total_chunks)
    
    print(f"Completed transforming text chunks into embeddings.\n\n")
    print("------------------COMPUTE EMBEDDINGS TRANSFORMATION END------------------")
    
    return df  # Corrected return statement here.

def search_docs(df, query, top_n=TOP_N):
    print("\n--------------------------------------------------------------------------------")
    print("\n------------------SEARCHING DOCS START------------------")
    print(f"Searching for top {TOP_N} relevant sentences in the document for the query: {query}")

    query_embedding = get_embedding_with_retries(query, os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT"), chunk_num=0, total_chunks=1)
    df['similarities'] = df['embeddings'].apply(lambda x: cosine_similarity(x, query_embedding))
    df_sorted = df.sort_values('similarities', ascending=False).head(top_n)

    print(f"Completed searching for top {TOP_N} relevant sentences in the document for your query: {query}")
    print("------------------SEARCHING DOCS END------------------")
    print("--------------------------------------------------------------------------------\n")

    print(f"*****RESULTS*****\n")
    print(f"The top {TOP_N} relevant sentences in the document for the query '{query}' are:")
    for i, row in df_sorted.iterrows():
        print("\n--------------------------------------------------------------------------------")
        print(f"Sentence: {row['text']}")
        print(f"Cosine Similarity: {str(row['similarities'])}")
    
    return df_sorted

df_text = pd.DataFrame(process_text(), columns=['text'])
df_text = compute_embeddings(df_text)

print(f"\nComputing vectors and embeddings for the QUERY:\n{USER_QUERY}\n")
searched_docs = search_docs(df_text, USER_QUERY)

def summarize():
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    estimated_cost = (token_count / 1000) * TOKEN_COST

    print("\n------------------SUMMARY------------------")
    print(f"Script Started At: {str(start_time)}")
    print(f"Script Finished At: {str(end_time)}")
    print(f"Script Run Time: {str(elapsed_time)}")
    print(f"Total Number of Tokens Processed: {token_count} tokens")
    print(f"Estimated OpenAI embeddings cost (as of 2023-Oct-21): ${estimated_cost:.4f}")
    print("\n------------------END------------------")

summarize()