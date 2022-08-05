from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import spacy
import nltk


def lemma(nlp, word):
    word = word.replace("_", " ")
    doc = nlp(word)
    return doc[0].lemma_.lower()


class Google(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        print("### Create Word2Vec model ###")
        self.model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    def check_word(self, word):
        if word == '':
            return False
        word = lemma(self.nlp, word)
        if word not in self.model:
            #print('{} is Not Found.'.format(word))
            return False
        return True

    def get_vector(self, word):
        word = lemma(self.nlp, word)
        if self.check_word(word) is False:
            return 0
        vector = self.model[word]
        print('{} : {}'.format(word, vector))
        return vector

    def get_similarity(self, word1, word2):
        #word1 = lemma(self.nlp, word1)
        #word2 = lemma(self.nlp, word2)
        if self.check_word(word1) is False or self.check_word(word2) is False:
            #print('no.')
            return None
        sim = self.model.similarity(word1, word2)
        #print("{} - {} : {}".format(word1, word2, sim))
        return sim

    def get_most_similar(self, word, topn):
        res = list()
        # word = lemma(self.nlp, word)
        if self.check_word(word) is False:
            return res
        similar_list = None
        try:
            similar_list = self.model.most_similar(word, topn=topn)
        except KeyError:
            print("Key Error: {}".format(word))
            return res
        # print("{}の類義語：{}個".format(word, len(similar_list)))
        # print(similar_list)
        for s in similar_list:
            res.append(lemma(self.nlp, s[0]))
            #print(lemma(self.nlp, s[0]))
        # print(res)f
        return res


if __name__ == '__main__':
    wv = Google()
    topn = 100
    # wv.get_most_similar('easy', topn)
    # wv.get_most_similar('simple', topn)
    # wv.get_most_similar('difficult', topn)
    # wv.get_most_similar('hard', topn)
    # wv.get_most_similar('dog', topn)
    # wv.get_most_similar('cat', topn)
    # wv.get_most_similar('popular')
    wv.get_similarity('easy', 'simple')
    voc = ['look', 'come', 'fit', 'coexist', 'imprint', 'work', '1', '2', '9']
    for i in range(len(voc)):
        for j in range(i+1, len(voc)):
            wv.get_similarity(voc[i], voc[j])
