import re
import nltk

stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(nltk.corpus.stopwords.words('english'))

# UDF for tokenization
def tokenize(text):
  # tokenization
  tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('Xx/')) > 2 and len(re.sub('d+', '', word.strip('Xx/'))))]
  tokens = map(str.lower, tokens)
  # stemming
  stems = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
  return stems