import pandas as pd
import tensorflow as tf
import nltk
import numpy as np
import pickle
from tabulate import tabulate
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Conv1D, GlobalMaxPooling1D, Embedding
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
from gensim.models.phrases import Phraser

def read_datasets():
    TRAIN_DATASET = "train.csv"
    TEST_DATA = "test.csv"
    TEST_LABELS = "test_labels.csv"
    REDUNDANT_FIELDS = ["id"]

    # Read in training dataset
    train_dataset = pd.read_csv(TRAIN_DATASET)

    # # Split training_data into x_train and y_train -- SAVE FOR LATER
    # x_train = training_data[DATA_FIELD]
    # y_train = training_data[LABEL_FIELDS]

    # Read in test data
    test_data = pd.read_csv(TEST_DATA)
    test_labels = pd.read_csv(TEST_LABELS)

    # Combine test data and labels into one data frame
    test_dataset = pd.concat([test_data, test_labels], axis=1)

    # Remove redundant id field from both datasets
    train_dataset = train_dataset.drop(columns=REDUNDANT_FIELDS)
    test_dataset = test_dataset.drop(columns=REDUNDANT_FIELDS)

    # Remove samples with labels containing -1 in test dataset, this 
    # is a place holder for samples that were not assigned labels.
    test_dataset = test_dataset.drop(test_dataset[(test_dataset.toxic == -1) |
                                                (test_dataset.severe_toxic == -1) |
                                                (test_dataset.obscene == -1) |
                                                (test_dataset.threat == -1) |
                                                (test_dataset.insult == -1) |
                                                (test_dataset.identity_hate == -1)].index)
    
    train_dataset = pd.read_pickle("balanced_dataset.pickle")

    return train_dataset, test_dataset

# Functions used in pre-processing
# Sorted by alphabetical order
# ---------------------------------------

# Function to clean comments in train dataset
# Params: pd dataframe - Training dataset
# Return: List - cleaned comments
def clean_data(train_dataset):
    # Remove punctuation
    regex_str = "[^a-zA-Z\s]"
    train_dataset['comment_text'] = train_dataset['comment_text'].replace(regex=regex_str, value="")

    # Remove extra whitespaces
    regex_space = "\s+"
    train_dataset['comment_text'] = train_dataset['comment_text'].replace(regex=regex_space, value=" ")

    # Strip whitespaces
    train_dataset['comment_text'] = train_dataset['comment_text'].str.strip()

    # Lowercase
    train_dataset['comment_text'] = train_dataset['comment_text'].str.lower()

    # Convert comment_text column into a list
    comment_list = train_dataset['comment_text'].tolist()

    return comment_list

# Function to create gensim ngrams with models
# Params: List - tokenized comments with stopwords removed
# Returns: List - tokens with bigrams
def gensim_ngrams(comment_token_stop):
    # Create Gensim n-grams
    return [bigram_model[word] for word in comment_token_stop]

# Function to create Gensim ngram models
# Add trigrams etc as needed
# Returns: Gensim bigram model
def gensim_ngrams_model():
    # Gensim N-grams
    # Create bigram model
    bigram = Phrases(comment_token, min_count=5, threshold=100)
    return Phraser(bigram)

# Function to get NLTK POS Tagger
# Params: Token
# Returns: Dict - POS tagger
def nltk_get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    # Convert NOTK to wordnet POS notations
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) # Default to noun if not found

# Function to use NLTK lemmatizer
# Params: List - tokenized comments with stopwords removed
# Returns: List - lemmatized tokens
def nltk_lemmatize(comment_token_stop):

    nltk.download('averaged_perceptron_tagger')
    comment_lemma = []
    lemmatizer = WordNetLemmatizer()

    for comment in comment_token_stop:
        temp = []
        temp.append([lemmatizer.lemmatize(word, pos=nltk_get_wordnet_pos(word)) for word in comment])
        comment_lemma += temp

    return comment_lemma

# Function to remove NLTK stopwords
# Params: List - cleaned comments
# Returns: List - cleaned comments with stopwords removed
def nltk_stopwords(comment_cleaned):
    # Stopwords in English only
    STOP_WORDS = set(stopwords.words('english'))
    # Remove stopwords
    comment_stop = [word for word in comment_cleaned if word not in STOP_WORDS]
    
    return comment_stop

# Function to tokenize comments using NLTK Word Tokenize
# Params: List - cleaned comments
# Returns: List - tokenized comments
def nltk_tokenize(text):
    return [word_tokenize(word) for word in text]

# Function to pickle final pre-processed data
# Params: 
    # List - tokens that have been fully pre-processed
    # Str - file name
# Output: Pickle file in directory/repo 
def save_pickle(preprocessed, file_name):
    pickle.dump(preprocessed, open("{0}.pickle".format(file_name),"wb"))

# --------------------------------------------
# Function for all pre-processing functions
# Intended to make it easier to modify when experimenting with different pre-processing methods
def preprocess_data(train_dataset):

    comment_cleaned = clean_data(train_dataset)
    # NLTK Tokenize
    # comment_token = nltk_tokenize(comment_cleaned)

    # Create gensim ngram model
    # Commented out because unused in base model
    # bigram_model = gensim_ngrams_model()

    # Remove NLTK stopwords
    comment_stop = nltk_stopwords(comment_cleaned)
    # NLTK Tokenize
    comment_token_stop = nltk_tokenize(comment_stop)

    # Create gensim bigrams
    # Commented out because unused in base model
    #gensim_bigrams = gensim_ngrams(comment_stop_token)

    # NLTK Lemmatization
    comment_lemma = nltk_lemmatize(comment_token_stop)

    save_pickle(comment_lemma, "comment_lemma")
    print("Pre-processed data is in the form of comment_lemma.pickle")

def build_model(num_words):
    EPOCHS = 30
    INIT_LR = 1e-3

    model = Sequential()

    model.add(Embedding(num_words, 128))
    model.add(Dropout(0.4))
    model.add(Conv1D(128, 7, padding="valid", activation="relu", strides=3))
    model.add(Conv1D(128, 7, padding="valid", activation="relu", strides=3))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    adam = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    
    model.compile(loss='binary_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])
    
    return model




def calculate_average_word_length(doc):
    # Construct a list that contains the word lengths for each DISTINCT word in the document
    vocab_lengths = [len(i) for i in set(doc)] # TODO 4
    # Find the average word type length
    avg_vocab_length = sum(vocab_lengths) / len(vocab_lengths) # TODO 5

    return avg_vocab_length

def plot_top_words(text, n=10):
    words = comment_lemma
    allWords = []
    for wordList in words:
        allWords += wordList
        fd = FreqDist(allWords)
        fd.plot(n)
            
def plot_top_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1
            
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    x,y=zip(*top)
    plt.bar(x,y)
    

def plot_top_ngrams_barchart(text, n=2):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams=_get_top_ngram(text,n)[:10]
    x,y=map(list,zip(*top_n_bigrams))
    sns.barplot(x=y,y=x)
    

def plot_top_non_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)
            
    sns.barplot(x=y,y=x)

# Heatmap code
#def heatmap(df, title):
#    plt.figure('heatmap', figsize=[10,10])
#    plt.title(title)
#    df_corr = df.corr()
#    #df_corr = np.triu(df_corr, k=1)
#    sns.heatmap(df_corr, vmax=0.6, square=True, annot=True, cmap='YlOrRd')
#    plt.yticks(rotation = 45)
#    plt.xticks(rotation = 45)
#    plt.show()
#
#heatmap(df_targets, 'Comment Type Heatmap')