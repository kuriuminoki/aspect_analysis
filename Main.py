from Load import load_dataset, adjusting
from Entropy import print_entropy, print_voc_entropy, print_cluster_entropy
from Similarity import print_sim_ave, print_sim_voc
from Score import print_pos_score

import numpy as np
import copy

AMAZON = False


def analysis(filename, aspects, pos_tags, ngram):
    df = load_dataset(filename)
    df = adjusting(df, aspects)
    # --Score
    print_pos_score("result/api/", aspects, pos_tags)
    # --Entropy
    # print_entropy(copy.deepcopy(df), aspects, pos_tags, ngram)
    #print_voc_entropy(df, aspects, pos_tags, ngram)
    #print_cluster_entropy(df, aspects, pos_tags, ngram)
    # --Similarity
    # print_sim_ave("result/api/", aspects)
    #print_sim_voc(aspects)


def main():
    print("-Aspect analysis.")
    if AMAZON is True:
        print("AMAZON")
        filename = "data/amazon_us_pc_training.csv"
        aspects = ['cost', 'community', 'compatibility', 'functional',
                   'looks', 'performance', 'reliability', 'usability']
        save_location = "result/amazon/"
        # --Other Settings
        sentiments = ['POS', 'NEG', 'NEU']
        pos_tags = ['ADJ', 'ADV', 'NOUN', 'VERB']
        ngram = 3
    else:
        # --API Data
        print("API")
        filename = "data/training.csv"
        aspects = ['community', 'compatibility', 'documentation',
                   'functional', 'performance', 'reliability', 'usability']
        save_location = "result/api/"
        # --Other Settings
        sentiments = ['POS', 'NEG', 'NEU']
        pos_tags = ['ADJ', 'ADV', 'NOUN', 'VERB']
        ngram = 3

    analysis(filename, aspects, pos_tags, ngram)


if __name__ == '__main__':
    main()
