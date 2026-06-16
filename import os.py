import os
import pandas as pd
import chromadb
from openai import OpenAI

# 1. Load your cleaned dataset
df = pd.read_csv("../data/CustomerFeedback.csv")

# 2. Read Azure OpenAI key from environment
api_key = os.getenv("AZURE_OPENAI_KEY")
if not api_key:
    raise ValueError("❌ AZURE_OPENAI_KEY not set in environment. Please set it before running.")

# 3. Initialize OpenAI client
client = OpenAI(api_key=api_key)

# 4. Initialize ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="customer_feedback")

# 5. Generate embeddings and store in ChromaDB
for idx, row in df.iterrows():
    review_id = str(idx)
    text = row["clean_review"]

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    embedding = response.data[0].embedding

    collection.add(
        ids=[review_id],
        documents=[text],
        embeddings=[embedding]
    )

print("✅ Embeddings generated and stored in ChromaDB")
