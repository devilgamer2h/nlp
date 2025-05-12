
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

text = "Python is Great!. NLP is fun 😂"
print("Original text :", text)

token = word_tokenize(text)
print("Step 1: Tokenization :", token)

filtered_tokens = [word for word in token if word.isalpha()]
print(filtered_tokens)

valid_tokens = [word for word in filtered_tokens if re.match(r"^[A-Za-z]+$",word)]
print(valid_tokens)

stop_words = set(stopwords.words('english'))
meaningfull_words = [word.lower() for word in valid_tokens if word.lower() not in stop_words]
print(meaningfull_words)

stemer = PorterStemmer()
stemmed_words = [stemer.stem(word) for word in meaningfull_words]
print(stemmed_words)

