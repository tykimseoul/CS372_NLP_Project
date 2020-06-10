import nltk
from bs4 import BeautifulSoup
import pandas as pd
from operator import itemgetter
from html.parser import HTMLParser
import re
import math

def linetoindex(pos, code):
    result  = []
    for (line, off) in code:
        result.append(pos[line-2] + off +1)
    return result

def full_index(pos, start, end, length):
    result = []
    start = linetoindex(pos, start)
    end = linetoindex(pos, end)
    for ind in range(len(start)):
        result.append(tuple((start[ind], end[ind] + 7)))
    final = []
    if len(result) == 1:
        final.append(tuple((0, result[0][0])))
        final.append(tuple((result[0][1], length)))
    elif len(result) ==2:
        final.append(tuple((0, result[0][0])))
        final.append(tuple((result[0][1], result[1][0])))
        final.append(tuple((result[1][1], length)))
    else:
        num = len(result)
        final.append(tuple((0, result[0][0])))
        for i in range(num-1):
            final.append(tuple((result[i][1], result[i+1][0])))
        final.append(tuple((result[num-1][1], length))) 
    return final
    

def exclude_code(document):
    parser = Sen_HTMLParser()
    parser.feed(document)
    start = parser.start
    end = parser.end
    parser.close
    if len(start) > 0 and len(start) == len(end):
        pos = [m.start() for m in re.finditer('\n', document)]
        slicing_index = full_index(pos,start,end, len(document))
        sliced = [document[s:e] for (s,e) in slicing_index]
        return "".join(sliced)
    else:
        return document

def strip_tags(document):
    soup = BeautifulSoup(document, "html.parser")
    text = soup.findAll(text=True, recursive=True)
    text = list(filter(lambda s: s!='[' and s!= '\n' and s!=']', text))
    spt = [re.split(r'[^a-zA-Z\']|\s+', sent) for sent in text]
    spt = [list(filter(lambda s: len(s) > 0, sent)) for sent in spt]
    return spt

class Sen_HTMLParser(HTMLParser):
    
    def __init__(self):
        self.reset()
        HTMLParser.__init__(self)
        self.start = []
        self.end = []
    def handle_starttag(self, tag, attrs):
        if tag == 'code':
            self.start.append(self.getpos())
    def handle_endtag(self, tag):
        if tag == 'code':
            self.end.append(self.getpos())
    def close():
        self.start = []
        self.end = []

def pos_tagger(document):
    return [[w for w in nltk.pos_tag(sent) if w[1].startswith('N') or w[1].startswith('J')] for sent in document]
def make_dict(document):
    return [ list(dict.fromkeys(tag_sent)) for tag_sent in document ]

def count_num(document):
    return [len(sent) for sent in document]

def sort_count(doc):
    indexed = [ tuple((i+1, doc[i])) for i in range(len(doc)) ]
    sorted_list = [i for (i, n) in sorted(indexed, key=lambda x:x[1], reverse = True)]
    return sorted_list[:5] if len(sorted_list)>5 else sorted_list
        

data = pd.read_csv('question_corpus.csv')
data = data[:300]
data['content'] = data['content'].map(lambda c: exclude_code(c))

data['sent'] = data['content'].map(lambda c: strip_tags(c))
data['num_sent'] = data['sent'].map(lambda c: count_num(c))

data['tagged'] = data['sent'].map(lambda s: pos_tagger(s))
data['num_tagged'] = data['tagged'].map(lambda c: count_num(c))

data['filtered'] = data['tagged'].map(lambda s: make_dict(s))
data['num_filtered'] = data['filtered'].map(lambda c: count_num(c))
data['sorted'] = data['num_filtered'].map(lambda c: sort_count(c))

data[['sent','filtered', 'sorted']].to_csv('output300.csv', index=False, header=True)

