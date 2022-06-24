import os
import numpy as np
import matplotlib.pyplot as plt

from constants import FOLDER_TEST_NEG, FOLDER_TEST_POS, FOLDER_TRAIN_NEG, FOLDER_TRAIN_POS
from utils import extract_rating_imdb, wordArray

def generate_imdb_stats():
    print('=============== Start ===============')
    plt_dist_imdb([FOLDER_TRAIN_NEG, FOLDER_TRAIN_POS], False)
    print("INFO: Plot produced to folder 'output'")
    plt_dist_imdb([FOLDER_TEST_NEG, FOLDER_TEST_POS ], True)
    print("INFO: Plot produced to folder 'output'")
    num = calcAvgWords([FOLDER_TEST_NEG, FOLDER_TEST_NEG, FOLDER_TRAIN_NEG, FOLDER_TEST_POS ])
    print('INFO: Average word count of one sample from imdb dataset: ' + str(round(num, 2)))
    print('INFO: Start to generate vocab')
    vocab = getVocab([FOLDER_TEST_NEG, FOLDER_TEST_NEG, FOLDER_TRAIN_NEG, FOLDER_TEST_POS ])
    np.savetxt('output/vocab_imdb.csv', np.asarray(vocab), fmt='%s')
    print('INFO: See Vocab for imdb dataset in file output/vocab.csv')
    print('INFO: Vocab contains ' + str(len(vocab)) + ' tokens.')
    print('================ Done ===============')


def getVocab(folderpaths):
    vocab = []
    folderCount = 0
    for folderepath in folderpaths:
        folderCount += 1
        fileCount = 0
        for filename in os.listdir(folderepath):
            fileCount += 1
            if fileCount % 100 == 0:
                print('INFO: Vocab for Folder ' + str(folderCount) + ' of ' + str(len(folderpaths)) + ' -- File ' + str(fileCount) + ' of ' + str(len(os.listdir(folderepath))))
            with open(os.path.join(folderepath, filename), 'r') as f:
                text = f.read()
                reviewWords = wordArray(text)
                for word in reviewWords:
                    if word not in vocab:
                        vocab.append(word)
    return vocab


def calcAvgWords(folderpaths):
    wordCount = 0
    docCount  = 0
    for folderepath in folderpaths:
        for filename in os.listdir(folderepath):
            with open(os.path.join(folderepath, filename), 'r') as f:
                text = f.read()
                wordCount += len(text.split())
                docCount += 1
    return wordCount / docCount


def plt_dist_imdb(folderpaths, train: bool):
    ratings = []
    for folderepath in folderpaths:
        for filename in os.listdir(folderepath):
            with open(os.path.join(folderepath, filename), 'r') as f:
                ratings.append(extract_rating_imdb(filename))
        hist = np.histogram(np.array(ratings, dtype=int), bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], height=hist[0])
        plt.title('Distribution of {s} data IMDB'.format(s = 'train' if train else 'test'))
        plt.savefig('output/dist_{s}_imdb.jpg'.format(s = 'train' if train else 'test'))
        plt.close()