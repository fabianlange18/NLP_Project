import os
import nltk
import random
import pickle

from constants import FOLDER_TEST_POS, FOLDER_TEST_NEG, FOLDER_TRAIN_POS, FOLDER_TRAIN_NEG
from utils import extract_class_imdb, wordArray

def readIMDBVocab():
    vocab = []
    with open('output/vocab_imdb.csv', 'r') as f:
        for line in f.readlines():
            vocab.append(line[:-1])
    return vocab

def wordCountFeature(document):
    documentWords = wordArray(document)
    vocab = readIMDBVocab()
    features = {}
    for word in vocab[:70000]:
        features['contains({})'.format(word)] = word in documentWords
    return features

def buildFeatureArrays(size):
    labeled_test = []
    labeled_train = []
    for test_folderpath in [FOLDER_TEST_POS, FOLDER_TEST_NEG]:
        for filename in sorted(os.listdir(test_folderpath))[:size]:
            with open(os.path.join(test_folderpath, filename), 'r') as f:
                labeled_test.append((f.read(), extract_class_imdb(filename)))
    for train_folderpath in [FOLDER_TRAIN_POS, FOLDER_TRAIN_NEG]:
        for filename in sorted(os.listdir(train_folderpath))[:size]:
            with open(os.path.join(train_folderpath, filename), 'r') as f:
                labeled_train.append((f.read(), extract_class_imdb(filename)))
    counter = 0
    for set in labeled_test, labeled_train:
        featureSet = [(wordCountFeature(text), label) for (text, label) in set]
        f = open('storage/{}.pickle'.format('test_set' if counter == 0 else 'train_set'), 'wb')
        pickle.dump(featureSet, f)
        f.close()
        print('{0} saved to storage/{0}.pickle'.format('test_set' if counter == 0 else 'train_set'))
        counter += 1

def fitModel():
    # Load train set
    f = open('storage/train_set.pickle', 'rb')
    train_set = pickle.load(f)
    f.close()

    random.shuffle(train_set)
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    # Save
    f = open('storage/imdb_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    print('Classifier stored to storage/imdb_classifier.pickle')
    f.close()


def evaluateModel():
    
    f = open('storage/imdb_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()

    f = open('storage/test_set.pickle', 'rb')
    test_set = pickle.load(f)
    f.close()

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    falseIndices = []

    for i, (feats, label) in enumerate(test_set):
        predicted = classifier.classify(feats)
        if label == 'pos':
            if predicted == 'pos':
                tp += 1
            else:
                fn += 1
                falseIndices.append(i)
        else:
            if predicted == 'neg':
                tn += 1
            else:
                fp += 1
                falseIndices.append(i)
    
    total = tp + tn + fp + fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    print("-" * 75)
    print('Model Statistics')
    print('Total predictions:   ' + str(total))
    print('Correct predictions: ' + str(tp + tn))
    print('False-positives:     ' + str(fp))
    print('False-negatives:     ' + str(fn))
    print('Accuracy:            ' + str((tp + tn) / total))
    print('Precision:           ' + str(precision))
    print('Recall:              ' + str(recall))
    print('F-Measure:           ' + str(2 * precision * recall / (precision + recall)))
    print("-" * 75)
    classifier.show_most_informative_features()
    print("-" * 75)
    
    print('Some imaginary examples')
    print('Predicting: What a great movie. I really like it. It was so much fun watching it with my family.')
    print('Result: ' + str(classifier.classify(wordCountFeature('What a great movie. I really like it. It was so much fun watching it with my family.'))))
    print('Predicting: This movie was so bad an boring, I really hated it. The plot is complete waste and the whole storyline was awful.')
    print('Result: ' + str(classifier.classify(wordCountFeature('This movie was so bad an boring, I really hated it. The plot is complete waste and the whole storyline was awful.'))))
    print("-" * 75)
    print("-" * 75)
    print('Some real examples that were classified correctly')
    print("-" * 75)
    size = int(total / 2)
    counter = 0
    for folder in [FOLDER_TEST_POS, FOLDER_TEST_NEG]:
        printCounter = 0
        for filename in sorted(os.listdir(folder))[:size]:
            if counter not in falseIndices and printCounter < 5:
                with open(os.path.join(folder, filename), 'r') as f:
                    text = f.read()
                    print('Predicting: ' + text)
                    print("-" * 18)
                    print('Correct Value: ' + extract_class_imdb(filename))
                    print('Model predicts the correct value')
                    print("-" * 75)
                    print("-" * 75)
                    printCounter += 1
            counter += 1
    print("-" * 75)
    print('Some real examples that were classified wrong')
    print("-" * 75)
    size = int(total / 2)
    counter = 0
    for folder in [FOLDER_TEST_POS, FOLDER_TEST_NEG]:
        printCounter = 0
        for filename in sorted(os.listdir(folder))[:size]:
            if counter in falseIndices and printCounter < 5:
                with open(os.path.join(folder, filename), 'r') as f:
                    text = f.read()
                    print('Predicting: ' + text)
                    print("-" * 18)
                    print('Correct Value: ' + extract_class_imdb(filename))
                    print('Model predicts the wrong value')
                    print("-" * 75)
                    print("-" * 75)
                    printCounter += 1
            counter += 1