
import nltk
from nltk import CFG

grammer = CFG.fromstring("""
  S -> NP VP
  NP -> Det N | Det Adj N | PN
  VP -> V NP | V
  V -> 'chased' | 'saw' | 'ate'
  Det -> 'the' | 'a'
  Adj -> 'small' | 'big'
  N -> 'cat' | 'dog' | 'man' | 'park'
  PN -> 'Mary' | 'John'
    """)

test_sentence = [
    "the cat chased the dog",
    "Mary ate",
    "John saw the dog",
    "the big dog saw the cat"
]

for sent in test_sentence:
  sentence = sent.split()
  print(f"\n === Parsing {''.join(sentence)} ===")
  print("\n**Top_down Parsing**")
  rd_parser = nltk.RecursiveDescentParser(grammer)
  found_parse = False
  for tree in rd_parser.parse(sentence):
    found_parse=True
    print(tree)
    tree.pretty_print()
  if not found_parse:
    print("no parse tree found!")

  print("\n**Bottom-Up Parsing**")
  chart_parser = nltk.ChartParser(grammer)
  found_parse = False
  for tree in chart_parser.parse(sentence):
    found_parse = True
    print(tree)
    tree.pretty_print()
  if not found_parse:
    print("no parse tree found!")

