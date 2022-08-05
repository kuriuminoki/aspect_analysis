from Load import load_voc, load_score
from Similarity import get_key


def print_pos_score(save_location, aspects, pos_tags):
    vocabulary = load_voc("pre_vocabulary.txt")
    score = load_score(save_location, aspects)
    word_sum = {}
    score_sum = {}
    for aspect in aspects:
        score[aspect] = score[aspect].set_index('word')
        word_sum[aspect] = {}
        score_sum[aspect] = {}
        for pos in pos_tags:
            word_sum[aspect][pos] = 0
            score_sum[aspect][pos] = 0

    for voc_name in vocabulary:
        pos, aspect, sentiment = get_key(voc_name)
        words = vocabulary[voc_name]
        for w in words:
            score_sum[aspect][pos] += score[aspect]['score'][w]
        word_sum[aspect][pos] += len(words)

    for aspect in aspects:
        print(aspect)
        for pos in pos_tags:
            s = score_sum[aspect][pos]
            n = word_sum[aspect][pos]
            print("  {}: sum={}, average={}, size={}".format(pos, s, s / n, n))
