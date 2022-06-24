def extract_rating_imdb(filename: str):
    length = len(filename)
    rating = int(filename[length - 5 : length - 4])
    return rating if rating != 0 else 10

def extract_class_imdb(filename: str):
    length = len(filename)
    rating = int(filename[length - 5 : length - 4])
    rating = rating if rating != 0 else 10
    return 'pos' if rating > 5 else 'neg'

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def wordArray(text):
    documentWords = []
    stopWords = set(stopwords.words('english'))
    tokenized = word_tokenize(text.lower())
    for word in tokenized:
        word_without_special_chars = ''.join(c for c in word if c.isalnum())
        if word_without_special_chars:
                if word_without_special_chars not in documentWords and word_without_special_chars not in stopWords:
                    documentWords.append(word_without_special_chars)
    return documentWords