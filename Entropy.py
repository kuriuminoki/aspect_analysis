from Load import load_voc
from Similarity import get_key

import random
import spacy
import math
import statistics


def get_ngram(doc, ind, ngram):
    """
    :param doc: Documents already analyzed by spaCy
    :param ind: First word
    :param ngram: Number of words to extract
    :return: res: Phrase
    """
    if ind + ngram > len(doc):
        return None
    res = doc[ind].lemma_.lower()
    cnt = 1
    while cnt < ngram and ind + cnt < len(doc):
        if doc[ind + cnt].pos_ == "PUNCT":
            return None
        res += ' ' + doc[ind + cnt].lemma_.lower()
        cnt += 1
    return res


def get_entropy(data_list, pos_tags, ngram, nlp):
    word_sum = 0
    word_dir = dict()
    for data in data_list:
        doc = nlp(data)
        for ind in range(len(doc)):
            if not(doc[ind].pos_ in pos_tags):
                continue
            for i in range(1, ngram + 1):
                word_sum += 1
                # --Get phrase
                phrase = get_ngram(doc, ind, i)
                word_dir[phrase] = word_dir.get(phrase, 0) + 1

    e = 0
    for phrase in word_dir:
        p = word_dir[phrase] / word_sum
        e += p * math.log2(p)
    e *= -1
    return e


def print_entropy(df, aspects, pos_tags, ngram):
    print("print entropy.")
    nlp = spacy.load('en_core_web_sm')

    # --count the size of data
    data_list = {}
    for aspect in aspects:
        data_list[aspect] = list()
    for i in range(len(df)):
        aspect_list = df['aspect'][i].split(',')
        for aspect in aspect_list:
            if aspect in aspects:
                data_list[aspect].append(df['sentence'][i])
    min_size = -1
    for aspect in aspects:
        if min_size == -1 or min_size > len(data_list[aspect]):
            min_size = len(data_list[aspect])

    # --get Entropy
    for aspect in aspects:
        ite = ((len(data_list[aspect]) + min_size - 1) // min_size)
        ite *= 100
        print("{} : size = {}, iteration = {}".format(aspect, len(data_list[aspect]), ite))
        e_sum = 0
        # --sampling min_size data and iteration is ite.
        for i in range(ite):
            samples = random.sample(data_list[aspect], min_size)
            e_sum += get_entropy(samples, pos_tags, ngram, nlp)
        e = e_sum / ite
        print('H({}) = {}\n'.format(aspect, e))


def print_voc_entropy(df, aspects, pos_tags, ngram):
    nlp = spacy.load('en_core_web_sm')
    word_cnt = dict()
    for aspect in aspects:
        word_cnt[aspect] = dict()
    for index in range(len(df)):
        doc = nlp(df['sentence'][index])
        aspect_list = df['aspect'][index].split(',')
        for ind in range(len(doc)):
            if not(doc[ind].pos_ in pos_tags):
                continue
            for i in range(1, ngram + 1):
                # --Get phrase
                phrase = get_ngram(doc, ind, i)
                for aspect in aspect_list:
                    if not(aspect in aspects):
                        continue
                    word_cnt[aspect][phrase] = word_cnt[aspect].get(phrase, 0) + 1

    #vocabulary = load_voc("vocabulary.txt")
    vocabulary = load_voc("pre_vocabulary.txt")
    entropy = dict()
    for aspect in aspects:
        entropy[aspect] = list()
    for voc_name in vocabulary:
        words = vocabulary[voc_name]
        if len(words) < 2:
            continue
        pos, aspect, sentiment = get_key(voc_name)
        cnt = 0
        for word in words:
            cnt += word_cnt[aspect][word]
        h = 0
        for word in words:
            p = word_cnt[aspect][word] / cnt
            h += p * math.log2(p)
        h *= -1
        entropy[aspect].append(h)
        print("H({}) = {}".format(voc_name, h))

    for aspect in aspects:
        mean = statistics.mean(entropy[aspect])
        print("H({}) = {}".format(aspect, mean))


def print_cluster_entropy(df, aspects, pos_tags, ngram):
    nlp = spacy.load('en_core_web_sm')
    word_cnt = dict()
    for aspect in aspects:
        word_cnt[aspect] = dict()
    for index in range(len(df)):
        doc = nlp(df['sentence'][index])
        aspect_list = df['aspect'][index].split(',')
        for ind in range(len(doc)):
            if not(doc[ind].pos_ in pos_tags):
                continue
            for i in range(1, ngram + 1):
                # --Get phrase
                phrase = get_ngram(doc, ind, i)
                for aspect in aspect_list:
                    if not(aspect in aspects):
                        continue
                    word_cnt[aspect][phrase] = word_cnt[aspect].get(phrase, 0) + 1

    vocabulary = load_voc("vocabulary.txt")
    entropy = dict()
    voc_entropy = dict()
    for aspect in aspects:
        entropy[aspect] = list()
    for voc_name in vocabulary:
        print(voc_name)
        words = vocabulary[voc_name]
        # if len(words) < 2:
        #     continue
        pos, aspect, sentiment = get_key(voc_name)
        voc_name = voc_name[:-1]
        cnt = 0
        for word in words:
            cnt += word_cnt[aspect][word]
        if not(voc_name in voc_entropy):
            voc_entropy[voc_name] = list()
        voc_entropy[voc_name].append(cnt)

    for voc_name in voc_entropy:
        if len(voc_entropy[voc_name]) < 2:
            continue
        h = 0
        cnt = sum(voc_entropy[voc_name])
        for n in voc_entropy[voc_name]:
            p = n / cnt
            h += p * math.log2(p)
        h *= -1
        pos, aspect, sentiment = get_key(voc_name)
        entropy[aspect].append(h)
        print("\nH({}) = {}".format(voc_name, h))
        print(voc_entropy[voc_name])

    for aspect in aspects:
        mean = statistics.mean(entropy[aspect])
        print("H({}) = {}".format(aspect, mean))
