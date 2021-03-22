import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Conv1D, GlobalMaxPooling1D, Embedding
import tensorflow as tf
from tabulate import tabulate

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
    return train_dataset, test_dataset

def preprocess_data():
    #TODO
    print('Not implemented yet')
    return None

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