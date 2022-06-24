from imdb_stats import generate_imdb_stats
from twitter_stats import generate_twitter_stats
from simple_model import buildFeatureArrays, fitModel, evaluateModel

def main():
    #generate_imdb_stats()
    #generate_twitter_stats()
    buildFeatureArrays(1000)
    fitModel()
    evaluateModel()

if __name__ == '__main__':
    main()