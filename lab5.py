# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np

def train_naive_bayes(docs, labels):
  vocab = set()
  word_counts = {"comedy":Counter(),"action":Counter()}
  class_counts =  {"comedy":0,"action":0}

  for doc,label in zip(docs,labels):
    words = doc.split(',')
    vocab.update(words)
    word_counts[label].update(words)
    class_counts[label] += 1

  vocab_size = len(vocab)
  total_docs = sum(class_counts.values())
  class_probs = {cls: np.log(class_counts[cls]/total_docs) for cls in class_counts}
  return word_counts,class_probs,vocab,vocab_size,class_counts

def compute_likelihood(word_counts,vocab_size,class_word_count,doc,smoothing=1):
  likelihood = 0
  for word in doc.split(','):
    freq_words = word_counts[word] + smoothing
    likelihood += np.log(freq_words/(class_word_count + smoothing * vocab_size))
  return likelihood

def predict_naive_bayes(word_counts,class_probs,vocab,vocab_size,class_counts,doc):
  scores = {}
  for cls in class_probs:
    class_word_count = sum(word_counts[cls].values())
    likelihood = compute_likelihood(word_counts[cls],vocab_size,class_word_count,doc)
    scores[cls] = class_probs[cls] + likelihood
  return max(scores,key=scores.get)

docs = [
    "fun,couple,love,love",
    "fast,furious,shoot",
    "couple,fly,fast,fun,fun",
    "furious,shoot,shoot,fun",
    "fly,fast,shoot,love"
]
labels = ["comedy","action","comedy","action","action"]

word_counts,class_probs,vocab,vocab_size,class_counts = train_naive_bayes(docs, labels)

doc_d = "fast,couple,shoot,fly"

predict = predict_naive_bayes(word_counts,class_probs,vocab,vocab_size,class_counts,doc_d)

print(predict)

