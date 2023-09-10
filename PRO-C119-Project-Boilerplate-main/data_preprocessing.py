# Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# to stem words
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# create an instance of class PorterStemmer

nltk.download('punkt')

# importing json lib
import json
import pickle
import numpy as np


words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
pattern_word_tags_list = [] #list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')
word_tags_list = []


# words to be ignored while creating Dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

# open the JSON file, load data from it.
train_data_file = open('C:/Users/Duvashi Family/Downloads/PRO-C119-Project-Boilerplate-main/PRO-C119-Project-Boilerplate-main/intents.json')
intents = json.load(train_data_file)
train_data_file.close()


# creating function to stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:

        # write stemming algorithm:
        '''
        Check if word is not a part of stop word:
        1) lowercase it 
        2) stem it
        3) append it to stem_words list
        4) return the list
        ''' 
        # Add code here #  
        # 
        # def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)      

    return stem_words


'''
List of sorted stem words for our dataset : 

['all', 'ani', 'anyon', 'are', 'awesom', 'be', 'best', 'bluetooth', 'bye', 'camera', 'can', 'chat', 
'cool', 'could', 'digit', 'do', 'for', 'game', 'goodby', 'have', 'headphon', 'hello', 'help', 'hey', 
'hi', 'hola', 'how', 'is', 'later', 'latest', 'me', 'most', 'next', 'nice', 'phone', 'pleas', 'popular', 
'product', 'provid', 'see', 'sell', 'show', 'smartphon', 'tell', 'thank', 'that', 'the', 'there', 
'till', 'time', 'to', 'trend', 'video', 'what', 'which', 'you', 'your']

'''

for intent in intents['intents']:
    
        # Add all words of patterns to list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            word_tags_list.append((pattern_word, intent['tag']))
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)

print(stem_words)

# creating a function to make corpus
def create_bot_corpus(stem_words, classes):

    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print(stem_words)
print(classes)

#Create Bag Of Words


#Create training data

training_data = []
number_of_tags = len(classes) 
labels = [0] *number_of_tags
for word_tags in word_tags_list: 
    bag_of_words = []
    pattern_words = word_tags[0] 
    for word in pattern_words: 
         index = pattern_words.index(word)
         word = stemmer.stem(word.lower())
         pattern_words[index] = word

    for word in stem_words: 
        if word in pattern_words:
              bag_of_words.append(1)
              
        else:
            bag_of_words.append(0)

    print(bag_of_words)
