import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
stop_words_list = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = re.findall(r'\w+', text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_list]
    return " ".join(tokens)
