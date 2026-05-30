import pandas as pd
df = pd.read_csv('gre_words_updated2.csv')
print(df[["word","synonyms"]].head(20))