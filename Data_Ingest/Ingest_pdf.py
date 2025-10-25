# Note: The first cell in the notebook was a pip install command.
# You may need to run this in your terminal before executing the script:
# pip install langchain_community langchain_sqlserver langchain_openai unstructured python-dotenv sqlalchemy pandas

# %%
# -------------------------
# 1. Imports & Environment
# -------------------------
import os
import re
import json
import pandas as pd
from dotenv import load_dotenv

# For PDF partitioning/extraction
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# Progress
from tqdm import tqdm
import numpy as np

# For embeddings
import openai
from langchain_openai import AzureOpenAIEmbeddings

# For SQL
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# For vector store
from langchain_sqlserver import SQLServer_VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import urllib.parse

# %%
# Load the .env file from the specified path
load_dotenv(override=True)

AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_EMBED = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')

connection_string = os.getenv('FABRIC_SQL_CONNECTION_URL_AGENTIC')

connection_url = f"mssql+pyodbc:///?odbc_connect={connection_string}"

engine = create_engine(connection_url, connect_args={"connect_timeout": 30})


# Setup Vector Store
# --- Instantiate your AzureOpenAIEmbeddings ---
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version="2024-10-21",
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# --- Build the VectorStore object ---
vector_store = SQLServer_VectorStore(
    connection_string=connection_url,           # same ODBC DSN used above
    distance_strategy=DistanceStrategy.COSINE,  # or DOT_PRODUCT, etc.
    embedding_function=embeddings,              # text-embedding-ada-002
    embedding_length=1536,                      # Vector dimension
    table_name="DocsChunks_Embeddings",         # Use the name you prefer
)

# Use AzureChatOpenAI for chat completions
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    openai_api_version="2024-10-21",
    openai_api_key=AZURE_OPENAI_KEY,
)

# %%
# ---------------
# 2. PDF Parsing
# ---------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Simple text-chunking utility.
    Splits the text into chunks of `chunk_size` characters
    with `overlap` characters overlap between chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # move the start back by overlap
        if start < 0:
            start = 0
    return chunks

pdf_path = "RAG_Preparation/SecureBank - Frequently Asked Questions.pdf"

elements = partition(pdf_path)
pdf_text = "\n".join([el.text for el in elements if el.text])  # combine into one big string

cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', str(pdf_text))
# cleaned = ' '.join(cleaned.split())
cleaned = cleaned.strip()

# Break into smaller chunks for embeddings
chunks = chunk_text(cleaned, chunk_size=500, overlap=100)

# %%
# Original notebook cell used: display(chunks)
# In a .py script, we'll use print() instead.
print("--- Displaying Chunks (Sample) ---")
print(chunks)
print(f"--- Total chunks created: {len(chunks)} ---")


# %%
# ----------------------------
# 3. Write raw chunks to SQL
# ----------------------------
# Suppose we create a table [PDF_RawChunks] with columns:
#   id INT IDENTITY(1,1) PRIMARY KEY
#   chunk_text NVARCHAR(MAX)
#   source_pdf NVARCHAR(512)  (optional, if you want to store PDF name/path)
#   created_at DATETIME2 DEFAULT GETDATE() (optional)
#
# Adjust as necessary if your table already exists.

create_table_sql = """
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[PDF_RawChunks]') AND type in (N'U'))
BEGIN
    CREATE TABLE [dbo].[PDF_RawChunks] (
        [id] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        [chunk_text] NVARCHAR(MAX) NOT NULL,
        [source_pdf] NVARCHAR(512) NULL,
        [created_at] DATETIME2 NOT NULL DEFAULT GETDATE()
    );
END
"""
with engine.begin() as conn:
    conn.execute(text(create_table_sql))


test_chunks = chunks[:200]  # comment this out later to process ALL chunks

batch_size = 100

with engine.connect() as conn:
    trans = conn.begin()
    for i, ck in enumerate(
        tqdm(test_chunks, desc="Inserting in 3x100 test batches", unit="chunk"),
        start=1
    ):
        conn.execute(
            text("INSERT INTO [dbo].[PDF_RawChunks] (chunk_text, source_pdf) VALUES (:ctext, :spdf)"),
            {"ctext": ck, "spdf": pdf_path}
        )

        # Commit every 100 inserts
        if i % batch_size == 0:
            trans.commit()
            # If there's more data left to insert, start a new transaction
            if i < len(test_chunks):
                trans = conn.begin()

    # If the total number isn't an exact multiple of batch_size
    # commit leftover rows in the final partial batch
    if len(test_chunks) % batch_size != 0:
        trans.commit()

print(f"Inserted {len(test_chunks)} chunks in batches of {batch_size}.")

# %%
# 1) Fetch Chunks from PDF_RawChunks That Need Embeddings
select_sql = """
SELECT RC.id, RC.chunk_text, RC.source_pdf
FROM PDF_RawChunks RC
WHERE NOT EXISTS (
    SELECT 1
    FROM DocsChunks_Embeddings VEC
    WHERE VEC.custom_id = CAST(RC.id AS VARCHAR(50))
)
ORDER BY RC.id
"""

with engine.connect() as conn:
    result = conn.execute(text(select_sql))
    rows = result.fetchall()

print(f"Found {len(rows)} row(s) in PDF_RawChunks with no existing embedding.")

# 2) Insert in Batches via vector_store.add_texts()

batch_size = 100  # commit in batches of 100

# Convert each row into text + metadata
all_texts = []
all_metadata = []

for row in rows:
    row_id   = row[0]
    text_val = row[1]
    pdf_path = row[2]

    # build your metadata
    meta_dict = {
        "custom_id": str(row_id),   # store the PDF_RawChunks ID as a string
        "source_pdf": pdf_path
    }

    all_texts.append(text_val)
    all_metadata.append(meta_dict)

print(f"Preparing to insert {len(all_texts)} texts in batches of {batch_size}...")

from math import ceil

num_rows = len(all_texts)
num_batches = ceil(num_rows / batch_size)

index = 0
for b in range(num_batches):
    # Slice out a batch
    batch_texts = all_texts[index : index+batch_size]
    batch_meta  = all_metadata[index : index+batch_size]
    index += batch_size

    print(f"Batch {b+1}/{num_batches}: inserting {len(batch_texts)} items...")
    
    # The vector_store call *immediately* does embeddings + inserts
    # into the underlying table. So each call is effectively a "mini commit."
    vector_store.add_texts(
        texts=batch_texts,
        metadatas=batch_meta
    )

print("All missing rows have been embedded and inserted into PDF_RawChunks_Embeddings!")

# %%
query = "What are the late payment fees on credit cards?"
docs = vector_store.similarity_search(query, k=3)
for i, doc in enumerate(docs, 1):
    print(f"\nResult {i}:\nMetadata = {doc.metadata}\nText    = {doc.page_content[:150]}...")