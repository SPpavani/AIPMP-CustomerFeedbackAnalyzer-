import os
from openai import OpenAI

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

client = OpenAI(
    api_key=api_key,
    base_url=f"{endpoint}openai/deployments/{deployment}",
    default_query={"api-version": "2023-05-15"},
    default_headers={"api-key": api_key}
)

response = client.embeddings.create(
    model=deployment,
    input="This is a test sentence for embeddings."
)

print("✅ Test embedding length:", len(response.data[0].embedding))
