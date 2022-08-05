import pandas as pd
import copy


# Load CSV dataset
def load_dataset(filename):
    df = pd.read_csv(filename, encoding='utf-8')
    # df = pd.read_csv(filename, encoding='cp932')
    return df


# Unify the notation of sentiment to POS, NEG and NEU
def sentiment_unification(df, aspects):
    res_df = copy.deepcopy(df)

    pos_list = ['positive', '1', 'Positive', 'positve', 'positive ', 'pos']
    neg_list = ['negative', '-1', 'Negative', 'neg']
    neu_list = ['0', 'negative/positive', 'discarded', 'neu']
    for index in range(len(df['sentiment'])):
        for index2 in range(len(pos_list)):
            if pos_list[index2] == df['sentiment'][index]:
                res_df['sentiment'][index] = 'POS'
                break
        for index2 in range(len(neg_list)):
            if neg_list[index2] == df['sentiment'][index]:
                res_df['sentiment'][index] = 'NEG'
                break
        for index2 in range(len(neu_list)):
            if neu_list[index2] == df['sentiment'][index]:
                res_df['sentiment'][index] = 'NEU'
                break
        if res_df['aspect'][index] in aspects and res_df['sentiment'][index] == 'NEU':
            res_df = res_df.drop(index)

    res_df.reset_index(drop=True, inplace=True)
    print("Remove data whose sentiment has NEU aspect")
    print("df: {} -> {}".format(len(df), len(res_df)))
    return res_df


# --Sentiment label is the same number as aspect label
def add_sentiment(df):
    for index in range(len(df)):
        included_aspects = df['aspect'][index].split(sep=",")
        included_sentiments = df['sentiment'][index].split(sep=",")
        gold_sentiment = ''
        for index2 in range(len(included_aspects)):
            if index2 > 0:
                gold_sentiment += ','
            if index2 < len(included_sentiments):
                gold_sentiment += included_sentiments[index2]
            else:
                gold_sentiment += included_aspects[0]
        df['sentiment'][index] = gold_sentiment
    return df


# --Filling NaN cells and Standardize notation
def adjusting(df, aspects):
    # --Fill NaN cells with none or NEU.
    df = df.fillna({'aspect': 'none'})
    df = df.fillna({'sentiment': 'NEU'})
    df = df.fillna({'sentence': 'ok'})

    # --Standardize notation to "POS, NEG, NEU"
    df = sentiment_unification(df, aspects)

    # --Add same number of sentiments as aspect
    df = add_sentiment(df)

    # --return train data and test data
    return df


def load_score(save_location, aspects):
    res = dict()
    for aspect in aspects:
        path = save_location + "word_score/" + aspect + ".csv"
        res[aspect] = pd.read_csv(path, encoding='utf-8')
    return res


def load_voc(path):
    res = dict()
    voc_file = open(path)
    for line in voc_file.readlines():
        text = line.strip()
        index = text.index(':')
        words = str(text[index + 2:-1]).split('|')
        res[str(text[:index])] = set(words)
    voc_file.close()
    return res
