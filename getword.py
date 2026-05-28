import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")
nltk.download("omw-1.4")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("gre_words.csv", encoding="utf-8-sig")

df["word"] = df["word"].astype(str).str.strip()


# =========================
# WORDNET SYNONYMS
# =========================
def get_synonyms(word):
    synonyms = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name)

    return list(synonyms)


# =========================
# WORDNET ANTONYMS
# =========================
def get_antonyms(word):
    antonyms = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace("_", " "))

    return list(antonyms)


# =========================
# SMART FALLBACK RULES (SCALE VERSION)
# =========================
def rule_based_antonym(word):
    word = word.lower()

    rules = {
        "in": "out",
        "im": "un",
        "ir": "rational",
        "il": "legal",
        "dis": "",
        "non": "",
    }

    # prefix-based guess (very useful for GRE words)
    for prefix, opposite_prefix in rules.items():
        if word.startswith(prefix):
            return word.replace(prefix, opposite_prefix, 1)

    return ""


# =========================
# FINAL ANTONYM ENGINE
# =========================
def smart_antonym(word):
    antonyms = get_antonyms(word)

    if antonyms:
        return ", ".join(antonyms[:5])

    fallback = rule_based_antonym(word)
    return fallback


# =========================
# SYNONYM ENGINE
# =========================
def smart_synonym(word):
    synonyms = get_synonyms(word)

    if synonyms:
        return ", ".join(synonyms[:8])

    return ""


# =========================
# APPLY TO DATASET
# =========================
df["synonyms"] = df["word"].apply(smart_synonym)
df["antonyms"] = df["word"].apply(smart_antonym)


# =========================
# SAVE RESULT
# =========================
df.to_csv("gre_words_updated2.csv", index=False)

print("DONE — processed 1000+ words successfully!")