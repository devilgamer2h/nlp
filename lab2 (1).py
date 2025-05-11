import nltk 
nltk.download('wordnet') 
nltk.download('omw-1.4')
from nltk.corpus import wordnet 
 
def get_synonyms_antonyms(word):
    synonyms = set() 
    antonyms = set() 
 
    for syn in wordnet.synsets(word): 
        for lemma in syn.lemmas(): 
            # Add synonym 
            synonyms.add(lemma.name()) 
             
            # Check and add antonym if exists 
            if lemma.antonyms(): 
                for ant in lemma.antonyms(): 
                    antonyms.add(ant.name()) 
 
    return synonyms, antonyms 
# Test word 
word = "active" 
synonyms, antonyms = get_synonyms_antonyms(word) 
 
print(f"Synonyms of '{word}':") 
print(", ".join(sorted(synonyms))) 
 
print(f"\nAntonyms of '{word}':") 
print(", ".join(sorted(antonyms))) 
