from Load import load_score, load_voc
from Word2vec import Google

import statistics


def get_key(voc_name):
    a = voc_name.split(sep='_')
    pos = a[0].upper()
    aspect = a[1].lower()
    sentiment = None
    if len(a) > 2:
        sentiment = a[2].upper()
    for i in range(10):
        aspect = aspect.replace(str(i), '')
        sentiment = sentiment.replace(str(i), '')
    return pos, aspect, sentiment


def get_sim_ave(words, wv):
    res = 0
    num = 0
    for i in range(len(words) - 1):
        for j in range(i, len(words)):
            s = wv.get_similarity(words[i], words[j])
            if s is None:
                continue
            res += s
            num += 1
    if num == 0:
        return 0
    res /= num
    return res


# --スコアが上位のK個で類似度平均を出す
def print_sim_ave(save_location, aspects):
    print("print similarity.")
    wv = Google()
    score = load_score(save_location, aspects)
    min_size = -1
    for aspect in aspects:
        if min_size == -1 or min_size > len(score[aspect]):
            min_size = len(score[aspect])
    for aspect in aspects:
        print(aspect)
        words = list()
        for i in range(len(score[aspect])):
            if i == min_size:
                break
            words.append(score[aspect]['word'][i])
        print(words)
        sim = get_sim_ave(words, wv)
        print("S({}) = {}\n".format(aspect, sim))


# --ボキャブラリごとの類似度平均を出す
def print_sim_voc(aspects):
    print("print vocabulary similarity.")
    wv = Google()
    #vocabulary = load_voc("vocabulary.txt")
    vocabulary = load_voc("pre_vocabulary.txt")
    result = dict()
    aspect_result = dict()
    for aspect in aspects:
        aspect_result[aspect] = list()
    for voc_name in vocabulary:
        print(voc_name)
        pos, aspect, sentiment = get_key(voc_name)
        if sentiment is None:
            continue
        words = list()
        for word in vocabulary[voc_name]:
            words.append(word)
        print(words)
        sim = get_sim_ave(words, wv)
        result[voc_name] = sim
        if sim > 0:
            aspect_result[aspect].append(sim)
        print("S({}) = {}\n".format(voc_name, sim))
    voc_rank = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(voc_rank)):
        print("{}: {} ({})".format(i + 1, voc_rank[i][0], voc_rank[i][1]))
        print("{}\n".format(vocabulary[voc_rank[i][0]]))

    for aspect in aspects:
        mean = statistics.mean(aspect_result[aspect])
        print("S({}) = {}".format(aspect, mean))
    # print(voc_rank)
