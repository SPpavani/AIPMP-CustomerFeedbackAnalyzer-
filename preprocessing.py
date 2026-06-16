import pandas as pd
import re

# Load dataset
df = pd.read_csv("../data/amazon_reviews.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special chars
    text = re.sub(r"\s+", " ", text)         # normalize spaces
    return text.strip()

df["clean_review"] = df["reviewText"].apply(clean_text)

# Save cleaned dataset
df.to_csv("../data/amazon_reviews_clean.csv", index=False)

print("Preprocessing complete. Cleaned file saved to data/amazon_reviews_clean.csv")
