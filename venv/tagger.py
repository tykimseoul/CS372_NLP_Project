from bs4 import BeautifulSoup
import pandas as pd
from operator import itemgetter
import re
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


def strip_tags(document):
    soup = BeautifulSoup(document, "html.parser")
    text = soup.findAll(text=True, recursive=True)
    spt = re.split(r'[^a-zA-Z\']|\s+', ' '.join(text))
    spt = list(filter(lambda s: len(s) > 0, spt))
    return spt


def tf(bag):
    bag = list(filter(lambda w: re.match('[a-zA-Z]+', w), bag))
    words = dict.fromkeys(set(bag), 0)
    for word in bag:
        words[word] += 1
    bag_size = len(bag)
    return dict(map(lambda w: (w[0], w[1] / bag_size), words.items()))


def idf(bag):
    bag = list(filter(lambda w: re.match('[a-zA-Z]+', w), bag))
    document_count = len(data)
    return dict(map(lambda w: (w, math.log(document_count / len(data[data.apply(lambda x: w in x['content'], axis=1)]))), bag))


def tf_idf(tf, idf, scores):
    words = sorted(list(map(lambda kv: (kv[0], kv[1] * idf[kv[0]]), tf.items())), key=itemgetter(1), reverse=True)
    if scores:
        return words
    else:
        return list(map(lambda w: w[0], words[:20]))


data = pd.read_csv('question_corpus.csv')
data = data[data.apply(lambda x: 'python' in x['tags'], axis=1)][:300]
data['content'] = data['content'].map(lambda c: strip_tags(c))
data['tf'] = data['content'].map(lambda c: tf(c))
data['idf'] = data['tf'].map(lambda w: idf(w.keys()))
data['tf-idf'] = data.apply(lambda df: tf_idf(df['tf'], df['idf'], False), axis=1)
data[['tags', 'tf-idf']].to_csv('tf_idf.csv', index=False, header=True)
print(data[['tags', 'tf-idf']].head(10))
