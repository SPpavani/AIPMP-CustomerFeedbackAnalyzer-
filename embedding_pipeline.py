import os
import pandas as pd
import chromadb
from openai import OpenAI
import re

# Load dataset
df = pd.read_csv("../data/CustomerFeedback.csv")

# Clean feedback
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_feedback"] = df["Feedback"].apply(clean_text)

# Read Azure environment variables
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")   # e.g. https://your-resource.openai.azure.com/
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g. text-embedding-ada-002

if not api_key or not endpoint or not deployment:
    raise ValueError("❌ Missing Azure OpenAI environment variables")

# Initialize client with Azure settings
client = OpenAI(
    api_key=api_key,
    base_url=f"{endpoint}openai/deployments/{deployment}",
    default_query={"api-version": "2023-05-15"},
    default_headers={"api-key": api_key}
)

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="customer_feedback")

# Generate embeddings
for idx, row in df.iterrows():
    review_id = str(row["CustomerID"])
    text = row["clean_feedback"]

    if not text:
        continue

    response = client.embeddings.create(
        model=deployment,   # use deployment name here
        input=text
    )
    embedding = response.data[0].embedding

    collection.add(
        ids=[review_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "Name": row["Name"],
            "Email": row["Email"],
            "Product": row["Product"],
            "Rating": row["Rating"],
            "Date": row["Date"]
        }]
    )

print("✅ Embeddings generated and stored in ChromaDB")
