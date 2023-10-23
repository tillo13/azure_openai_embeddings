# Azure OpenAI Embeddings & Weaviate - Ingesting Large PDF Documents

## Table of Contents
- [Project Overview](#project-overview)
- [Repo Structure](#repo-structure)
- [System Requirements](#system-requirements)
- [Setup & Configuration](#setup--configuration)
- [Running the Scripts](#running-the-scripts)
- [Repurposing for Other Documents](#repurposing-for-other-documents)
- [Potential Next Steps](#potential-next-steps)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The primary goal of these scripts is to utilize Azure OpenAI's embeddings/vector and language model, Ada V2 to create embeddings from ingesting a PDF.  I wanted to test both local embedding values and also compare to the vector database capabilities of Weaviate to process.

Anticipating a variety of challenges in the text data, I used the novel "The Hobbit" as the sample PDF file. It's complex, detailed, and reflective of real-life scenarios. By breaking down textual data into manageable chunks and transforming each into a vector representation ("embedding"), these scripts allow not just easy processing of large text, but also a convenient way to access and analyze specific parts of your document.

## Repo Structure

This repository contains two main Python scripts - `ingest_pdf_azure_openai_embeddings.py` and `ingest_pdf_azure_weaviate_openai_embeddings.py`. The former shows how to use Azure OpenAI's Language Model to grab insights from a large document, while the latter further implements a storage mechanism for embeddings using Weaviate, a highly scalable, graph-based vector search engine --testing out which works better, and sharing if you feel either work better for your use case.

Here is a screenshot of how Weaviate parses and gleans embeddings from `ingest_pdf_azure_weaviate_openai_embeddings.py`:

![Weaviate Demonstration](/images/weaviate_demo.png)

Here is a screenshot of using local embedding vectors/responses running the same type from: `ingest_pdf_azure_weaviate_openai_embeddings.py`:

![Script Results](/images/results.png)

## Setup & Configuration

Clone this repository and install necessary dependencies via `pip install -r requirements.txt`. The scripts contain global variables that allow fine-tuning of their operation to meet your specific needs. Refer to source files for more specifics.

## Environment Variables

Ensure the following keys are defined in your local `.env` file: 

- `OPENAI_EMBEDDINGS_DEPLOYMENT`: Azure OpenAI's Language Model (Ada V2).
- `OPENAI_EMBEDDINGS_API_KEY`: Your Azure OpenAI API key.
- `OPENAI_EMBEDDINGS_ENDPOINT`: Azure OpenAI's endpoint URL.
- `OPENAI_API_VERSION`: The version of Azure OpenAI's API you're using.

## Running the Scripts

After setting up, invoke a script with `python script.py` in your terminal. The console will keep you updated on the embeddings fetching, processing, and relevant sentences extracted based on the user query.

Below is an example of the terminal output while getting embeddings:

![Getting Embeddings](/images/getting_embeddings.png)

## Repurposing for Other Documents

These explorations are not restricted to "The Hobbit." Similar to the processes highlighted, any large PDF document can be processed. Set your query, and let the embeddings do the talking.

## Potential Next Steps

While the current implementations serve their purpose, there is always room for expansion and potential next steps:

1. **Comparison Study**: Conduct split tests and compare results for Azure OpenAI and Weaviate. This can provide insights on cases where one performs better than the other.

2. **Extending to Other Databases**: The current setup uses Weaviate. We could try implementing other vector databases like Milvus or Pinecone to see if they offer advantages in certain scenarios.

3. **Performance Optimization**: While chunking and vectorizing have been implemented, the performance still largely depends on your computing resources. We can explore techniques to further optimize the processing of large documents.

4. **User Interface**: A user interface can be developed for non-technical people to use these scripts easily, thereby making the tool more accessible.

5. **Complex Queries**: Queries can be more complex and structured, and more advanced NLP techniques can be used to capture the semantics of the question better.

## License

This project comes with an MIT License. You're free to modify, distribute, and make use of the code for your purposes, provided attributions remain in place.