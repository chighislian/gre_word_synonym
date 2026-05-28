import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")

# Load CSV correctly
df = pd.read_csv("gre_words.csv", encoding="utf-8-sig")

# Remove empty column if present
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])

print(df.columns)

# Function to get synonyms
def get_synonyms(word):

    synonyms = set()

    for syn in wn.synsets(str(word)):
        for lemma in syn.lemmas():
            synonyms.add(
                lemma.name().replace("_", " ")
            )

    synonyms.discard(str(word))

    return ", ".join(list(synonyms)[:5])

# Function to get antonyms
def get_antonyms(word):

    antonyms = set()

    for syn in wn.synsets(str(word)):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.add(
                    lemma.antonyms()[0]
                    .name()
                    .replace("_", " ")
                )

    return ", ".join(list(antonyms)[:5])

# Generate columns
df["synonyms"] = df["word"].apply(get_synonyms)
df["antonyms"] = df["word"].apply(get_antonyms)

# Save updated CSV
df.to_csv("gre_words_updated.csv", index=False)

print("Done! Updated CSV saved.")