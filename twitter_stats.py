import os
import numpy as np
import matplotlib.pyplot as plt
import csv

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from constants import FILE_TEST, FILE_TRAIN

def generate_twitter_stats():
    print('=============== Start ===============')
    plt_dist_twitter_test(FILE_TEST)
    print("INFO: Plot produced to file 'output'")
    num = calcAvgWords([FILE_TEST, FILE_TRAIN])
    print('INFO: Average word count of one sample from twitter dataset: ' + str(round(num, 2)))
    print('INFO: Start to generate vocab')
    vocab = getVocab([FILE_TEST, FILE_TRAIN])
    np.savetxt('output/vocab_twitter.csv', np.asarray(vocab), fmt='%s')
    print('INFO: See Vocab for twitter dataset in file output/vocab_twitter.csv')
    print('INFO: Vocab contains ' + str(len(vocab)) + ' tokens.')
    print('================ Done ===============')


def getVocab(filepaths):
    vocab = []
    stopWords = set(stopwords.words('english'))
    for filepath in filepaths:
        with open(filepath, 'r', encoding='latin-1') as f:
                csv_file = csv.reader(f, delimiter=',')
                for row in csv_file:
                    tokenized = word_tokenize(row[5].lower())
                    for word in tokenized:
                        word_without_special_chars = ''.join(c for c in word if c.isalnum())
                        if word_without_special_chars:
                            if word_without_special_chars not in vocab and word_without_special_chars not in stopWords:
                                vocab.append(word_without_special_chars)
                                print("INFO: Added " + word_without_special_chars + " to vocab for twitter dataset.")
    return vocab


def calcAvgWords(filepaths):
    wordCount = 0
    docCount  = 0
    for filepath in filepaths:
        with open(filepath, 'r', encoding='latin-1') as f:
            csv_file = csv.reader(f, delimiter=',')
            for row in csv_file:
                wordCount += len(row[5].split())
                docCount += 1
    return wordCount / docCount


def plt_dist_twitter_test(filepath):
    ratings = []
    with open(filepath, 'r', encoding='latin-1') as f:
        csv_file = csv.reader(f, delimiter=',')
        for row in csv_file:
            ratings.append(row[0])
    hist = np.histogram(np.array(ratings, dtype=int), bins=[0, 2, 4, 6])
    plt.bar([0, 2, 4], height=hist[0])
    plt.title('Distribution of test data Twitter')
    plt.savefig('output/dist_test_twitter.jpg')
    plt.close()