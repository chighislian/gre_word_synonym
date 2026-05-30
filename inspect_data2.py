import pandas as pd

df = pd.read_csv("gre_words_updated2.csv")

df["combined"] = df["word"].fillna("") + " " + df["synonyms"].fillna("")

matches = df[
    df["combined"].str.contains("bravery", case=False, na=False)
]

print(matches[["word", "synonyms"]])