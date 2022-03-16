import pandas as pd 
import re
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
nltk.download('words')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

#stop_words = stopwords.words('english')
stop_words = set(stopwords.words('english'))
english_words = set(nltk.corpus.words.words())


twitter_dataset = pd.read_csv("C:/Users/rulij/Downloads/archive/hashtag_joebiden.csv")
#twitter_dataset = pd.read_csv("C:/Users/rulij/Downloads/archive/hashtag_joebiden.csv", dtype={'text':str})

twitter_dataset = twitter_dataset.head(100)

# Translating emojis's meanings
emojis = {':)': 'smile', ':-)': 'smile', ';d':'wink', ':-E': 'vampire', ':(': 'sad',
':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked',
':-$': 'confused', '$_$':'greedy', '@@': 'eyeroll', ':-!':"confused", ':-D': 'smile', ':-0': 'yell',
'O.o': 'confused', '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-": 'sadsmile', ';)': 'wink', ';-)': 'wink',
'O:-)':'angel', 'o*-):':'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

def clean_data(data):
    data = str(data).lower()
    data = re.sub(r"@\S+ ", r'', data)

    for emoji in emojis.keys():
        data = data.replace(emoji,emojis[emoji])
    data = re.sub("\s+", ' ', data)
    data = re.sub("\n", ' ', data)
    data = re.sub("http\S+", " ", data)
    letters = re.sub("[^a-zA-Z]", " ", data)

    return letters

def stop_words_detector(words):
    filter_words = []
    filter_words_english = []

    for w in words:         
        if w not in stop_words:
            filter_words.append(w)  

    for x in filter_words:
        if x in english_words:
            filter_words_english.append(x) 

    return filter_words


    

#clean the data
twitter_dataset['tweet'] = twitter_dataset['tweet'].apply(lambda x:clean_data(x))
#splitting text into lists:
twitter_dataset['tweet'] = twitter_dataset['tweet'].apply(lambda x:x.split(" "))
#removing stop words:
twitter_dataset['tweet'] = twitter_dataset['tweet'].apply(lambda x:stop_words_detector(x))

# stemming
lemmatizer = WordNetLemmatizer()
twitter_dataset['tweet'] = twitter_dataset['tweet'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
# joining the list back to text
twitter_dataset['tweet'] = twitter_dataset['tweet'].apply(lambda x: ' '.join(x))
# # temp_data = twitter_dataset.head(100)
twitter_dataset.to_csv("C:/Users/rulij/Desktop/Intelligent system project/temp_data6.csv", sep = ',')

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.33)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

print("Done!")
