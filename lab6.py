

import nltk
from nltk.corpus import brown, inaugural, reuters, udhr, treebank, words
from nltk.corpus import PlaintextCorpusReader
from nltk import ConditionalFreqDist, UnigramTagger, DefaultTagger
from collections import Counter
import os

nltk.download('all')

print("BROWN:", brown.categories()[:5], brown.words(categories='news')[:10])
print("INAUGURAL:", inaugural.fileids()[:2], inaugural.words('2009-Obama.txt')[:10])
print("REUTERS:", reuters.categories()[:3], reuters.words(categories='crude')[:10])
print("UDHR:", udhr.fileids()[:2], udhr.words('English-Latin1')[:10])

os.makedirs('my_corpus', exist_ok=True)
with open('my_corpus/sample.txt', 'w') as f:
  f.write("This is a custom corpus. Testing custom data.")
custom = PlaintextCorpusReader('my_corpus', '.*\.txt')
print("Custom Corpus Words:", custom.words())

pairs = [(word.lower(), cat) for cat in brown.categories() for word in brown.words(categories=cat)]
cfd = ConditionalFreqDist(pairs)
print("CFD example (word 'news'):", cfd['news'].most_common(3))

print("Tagged Sents:", treebank.tagged_sents()[:1])
print("Tagged Words:", treebank.tagged_words()[:5])

tags = [tag for _, tag in treebank.tagged_words()]
noun_tags = [tag for tag in tags if tag.startswith('NN')]
print("Top NOUN Tags:", Counter(noun_tags).most_common(3))

props = {'cat': {'type': 'animal'}, 'apple': {'type': 'fruit'}}
for w, p in props.items():
  print(f"{w} → {p}")

default_tagger = DefaultTagger('NN')
unigram_tagger = UnigramTagger(treebank.tagged_sents()[:3000], backoff=default_tagger)
print("UnigramTagger Accuracy:", unigram_tagger.evaluate(treebank.tagged_sents()[3000:]))

wordlist = set(words.words())
def segment(text, min_len=2):
  results = []
  def backtrack(t, path=[]):
    if not t:
      results.append(path)
      return
    for i in range(min_len, len(t)+1):
      w = t[:i]
      if w in wordlist:
        backtrack(t[i:], path + [w])
  backtrack(text)
  return sorted([(seg, len(seg)) for seg in results],key=lambda x: x[1])
segments = segment("themanrantosave")
print("Segmented Words:")
for s, sc in segments: print(" →",  " ".join(s), "| Count:", sc)

