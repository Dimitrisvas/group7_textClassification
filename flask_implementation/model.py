import functools
import joblib
import nltk
import numpy as np
import pandas as pd
import pickle
import sklearn
import tensorflow as tf

from functools import lru_cache
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from os.path import join, dirname, realpath
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

# File paths

# Multinomial Naive Bayes model file path
MODEL_DIR = join(dirname(realpath(__file__)), "multi_mnb_model.pkl")

# Balanced datasets
BALANCED_TRAIN_DATASET = join(dirname(realpath(__file__)), "../balanced_dataset.pickle")
BALANCED_TEST_DATASET = join(dirname(realpath(__file__)), "../balanced_test_dataset.pickle")

# Preprocessed balanced data
PREPROCESSED_BAL_TRAIN_DATASET = join(dirname(realpath(__file__)), "../preprocessed_train.pickle")
PREPROCESSED_BAL_TEST_DATASET = join(dirname(realpath(__file__)), "../preprocessed_test.pickle")

# Function to load pickle file
# Params:
    # Str - @file_path: File path of pickle file
# Output:
    # Saved object in original file type (list/dataframe)
def load_pickle(file_path):
    return pickle.load(open(file_path, "rb"))

# Dummy function for TfidfVectorizer tokenizer
def fake_function(comments):
    return comments

# Pre-processing functions


# Function to clean comments in dataset
# Params: 
#   Pandas dataframe - @dataset: Data to be cleaned
# Output: 
#   List    - @comment_list: Cleaned comments (2D List)
def clean_data(dataset):

    # Remove punctuation
    regex_str = "[^a-zA-Z\s]"
    dataset['comment_text'] = dataset['comment_text'].replace(regex=regex_str, value="")

    # Remove extra whitespaces
    regex_space = "\s+"
    dataset['comment_text'] = dataset['comment_text'].replace(regex=regex_space, value=" ")

    # Strip whitespaces
    dataset['comment_text'] = dataset['comment_text'].str.strip()

    # Lowercase
    dataset['comment_text'] = dataset['comment_text'].str.lower()

    # Convert comment_text column into a list
    comment_list = dataset['comment_text'].tolist()

    return comment_list

# Function to get NLTK POS Tagger
# Params: 
#   Str - @word: Token
# Output
#   Dict - POS tagger
def nltk_get_wordnet_pos(word):
    
    tag = nltk.pos_tag([word])[0][1][0].upper()

    # Convert NLTK to wordnet POS notations

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN) # Default to noun if not found

# Function to use NLTK lemmatizer
# Params: 2D List - Tokenized comments with stopwords removed
# Returns: 2D List - lemmatized tokens
def nltk_lemmatize(comment_stop):

    nltk.download('averaged_perceptron_tagger')
    comment_lemma = []
    lemmatizer = WordNetLemmatizer()
    lemmatizer_cache = lru_cache(maxsize=50000)(lemmatizer.lemmatize)

    for comment in comment_stop:
        temp = []
        temp.append([lemmatizer_cache(word, pos=nltk_get_wordnet_pos(word)) for word in comment])
        comment_lemma += temp

    return comment_lemma

# Function to remove NLTK stopwords
# Params: 
#   2D List - @comment_token:   cleaned & tokenized comments
# Output:
#   2D List - @comment_stop: cleaned tokens with stopwords removed
def nltk_stopwords(comment_token):
    # Stopwords in English only
    STOP_WORDS = set(stopwords.words('english'))

    # Remove stopwords
    comment_stop = []

    for comment in comment_token:
        
        temp_word = []

        for word in comment:
            
            if word not in STOP_WORDS:
                temp_word.append(word)

        comment_stop.append(temp_word)

    return comment_stop

# Function to tokenize comments using NLTK Word Tokenize
# Params: 
#   2D List - @text: cleaned comments
# Output: 
#   2D List - tokenized comments
def nltk_tokenize(text):
    return [word_tokenize(word) for word in text]

# Function for all pre-processing functions without saving as pickle file
# Params:
#   List  - @dataset: Dataset to be pre-processed (train/test)
# Output:
#   List - @comments_list: Preprocessed tokens (2D List)
def preprocess_data_without_pickle(dataset):

    # Prevent re-running on already preprocessed data
    if isinstance(dataset, pd.DataFrame): #if dataframe, data isn't preprocessed

        comments_list = clean_data(dataset)
        
        # NLTK Tokenize
        comments_list = nltk_tokenize(comments_list)

        # Remove NLTK stopwords
        comments_list = nltk_stopwords(comments_list)

        # NLTK Lemmatization
        comments_list = nltk_lemmatize(comments_list)
        
        return comments_list
    
    else:
        return dataset

# Function to get dataset
# Output:
#   2D List - @bal_train_dataset: Training data
#   2D List - @bal_train_y      : Labels for training data
def get_dataset():
    # Get preprocessed train dataset
    bal_train_dataset = load_pickle(PREPROCESSED_BAL_TRAIN_DATASET)

    # Get train_y
    bal_train_y = pd.read_pickle(BALANCED_TRAIN_DATASET)
    bal_train_y = bal_train_y.drop(columns="comment_text")

    return bal_train_dataset, bal_train_y

# Function to build pipeline
# Output:
#   sklearn Model Pipeline - @pipe: Pipeline containing TFIDFVectorizer and model
def build_pipeline():

    # Build pipeline with TFIDF Vectorizer
    # Pass in dummy function into tokenizer
    # Pass in our custom preprocess function
    # Create Multinomial Naive Bayes MultiOutputClassifier model
    pipe = Pipeline([ 
    ('tfidf', TfidfVectorizer(
        analyzer='word', 
        tokenizer=fake_function, 
        preprocessor=preprocess_data_without_pickle, 
        token_pattern=None,
        min_df=5, 
        norm='l2', 
        smooth_idf=True, 
        sublinear_tf=True)), 
    ('multi_mnb', MultiOutputClassifier(MultinomialNB(), n_jobs=-1))
    ])

    return pipe

if __name__ == '__main__':
    # Get training data
    train_x, train_y = get_dataset()
    # Build pipeline with model and TFIDFVectorizer
    pipe = build_pipeline()
    # Fit pipeline
    pipe.fit(train_x, train_y)
    # Save model
    joblib.dump(pipe, 'flask_implementation/multi_mnb_model_test.joblib', compress=1)