
from collections import Counter
import pandas as pd

sentences = [
    "A Quick brown fox jumps over a lazy dog",
    "A journey of thousand miles start with a single step",
    "All glitter are not gold",
    "To be or not to be that is the question",
    "An apple a day keeps a doctor away"
]

unigrams = []
bigrams = []
trigrams = []

def tokenize(sentence):
  return sentence.lower().split()

def generate_ngram(tokens, n):
  return [tuple(tokens[i:i+n] ) for i in range(len(tokens)-n+1)]

for sentence in sentences:
  tokens = tokenize(sentence)
  unigrams.extend(generate_ngram(tokens, 1))
  bigrams.extend(generate_ngram(tokens, 2))
  trigrams.extend(generate_ngram(tokens, 3))

unigram_count = Counter(unigrams)
bigram_count = Counter(bigrams)
trigram_count = Counter(trigrams)

def calculate_ngram_probability(ngram_counts, lower_order_counts=None):
  probabilities = {}
  for ngram , count in ngram_counts.items():
    if lower_order_counts:
      prefix = ngram[:-1]
      prefix_count = lower_order_counts[prefix] if prefix in lower_order_counts else 1
      probabilities[ngram] = count / prefix_count
    else:
      probabilities[ngram] = count / sum(ngram_counts.values())
  return probabilities

unigram_probs = calculate_ngram_probability(unigram_count)
bigram_probs = calculate_ngram_probability(bigram_count,unigram_count)
trigram_probs = calculate_ngram_probability(trigram_count,bigram_count)

df_unigram = pd.DataFrame(unigram_probs.items(), columns = ["unigram", "Probability"])
df_bigram = pd.DataFrame(bigram_probs.items(), columns = ["bigram", "Probability"])
df_trigram = pd.DataFrame(trigram_probs.items(), columns = ["trigram", "Probability"])

print(df_unigram.to_string(index=False) )
print(df_bigram.to_string(index=False))



