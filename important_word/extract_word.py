import nltk
import pandas as pd
import re
import csv
import math
from functools import cmp_to_key
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer 

num_documents = 20

def load_document(raw_text):
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.findAll(text=True, recursive=True)
    spt = re.split(r'[^a-zA-Z\']|\s+', ' '.join(text))
    spt = list(filter(lambda s: len(s) > 0, spt))
    combined_text = ' '.join(spt)
    sentences = combined_text.split('.')
    tokenizer = RegexpTokenizer("[\w']+")
    processed_sentence_list = list()
    for sentence in sentences:
        tokenized_words = tokenizer.tokenize(sentence)
        tokenized_words = [w.lower() for w in tokenized_words if is_valid_word(w.lower())]
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in tokenized_words]
        processed_sentence_list.append(lemmatized_words)
    return processed_sentence_list

def preprocess(sentence):
    
    return nltk.pos_tag(sentence) 

def build_word_dict(document):
    word_dict = dict()
    word_dict['Noun'] = list()
    word_dict['Verb'] = list()
    word_dict['Adj'] = list()
    word_dict['Adv'] = list()
    word_dict['Det'] = list()
    for sentence in document:
        sentence_with_tag = nltk.pos_tag(sentence)
        for word, tag in sentence_with_tag:
            if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
                if word not in word_dict['Noun']:
                    word_dict['Noun'].append(word)
            elif tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
                if word not in word_dict['Verb']:
                    word_dict['Verb'].append(word)
            elif tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                if word not in word_dict['Adv']:
                    word_dict['Adv'].append(word)
            elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                if word not in word_dict['Adj']:
                    word_dict['Adj'].append(word)
            else:
                if word not in word_dict['Det']:
                    word_dict['Det'].append(word)

    return word_dict

def build_grammar(word_dict):
    grammar = '''
    S -> NP NP | NP VP | VP NP | S NP | NP S | S S | S Det | Det S | VP VP | VP S | S VP | N S | S N | Adv S | Adj S | S Adj | S Adv | Adv S
    S -> S N | N S | S V | V S | NP | VP
    NP -> N NP | Det N | NP N | Det NP | NP NP | Adj NP | N N | N Det | Adj N | NP Det
    VP -> V NP | V S | V NP PP | Det VP | VP VP | Adv VP | V Det | Det V | V VP | V V | VP Det
    Det -> Det Det | Adj Det
    Adv -> Adv Adv | Det Adv
    '''

    noun_list = ["'" + w + "'" for w in word_dict['Noun']]
    verb_list = ["'" + w + "'" for w in word_dict['Verb']]
    adjective_list = ["'" + w + "'" for w in word_dict['Adj']]
    adverb_list = ["'" + w + "'" for w in word_dict['Adv']]
    det_list = ["'" + w + "'" for w in word_dict['Det']]


    nouns = "|".join(noun_list)
    verbs = "|".join(verb_list)
    adjectives = "|".join(adjective_list)
    adverbs = "|".join(adverb_list)
    dets = "|".join(det_list)

    grammar += "\n N ->" + nouns
    grammar += "\n V ->" + verbs
    grammar += "\n Adj ->" + adjectives
    grammar += "\n Det ->" + dets
    grammar += "\n Adv ->" + adverbs

    return grammar

def get_depth(tree, word, depth = 1):
    greatest_depth = depth
    for subtree in tree:
        if type(subtree) is nltk.tree.Tree:
            analyzed_depth = get_depth(subtree, word, depth + 1) 
            if analyzed_depth >= greatest_depth:
                greatest_depth = analyzed_depth
        else:
            if "'" + subtree + "'" == word:
                print("leave : {}, depth : {}".format(subtree, depth))
                return depth + 1
            else:
                return -1
    return greatest_depth


def get_avg_depth_dict(document, grammar):
    print(grammar)
    parser = nltk.ShiftReduceParser(nltk.CFG.fromstring(grammar), trace = 1)
    depth_dict = dict()
    for sentence in document:
        tree = parser.parse(sentence)
        for word in sentence:
            depth = get_depth(tree, word)
            if word in depth_dict.keys():
                freq, prev_depth = depth_dict[word]
                depth_dict[word] = (freq + 1, prev_depth + 1)
            else:
                depth_dict[word] = (1, depth)

    avg_depth_dict = dict()
    for word, (freq, depth_sum) in depth_dict.items():
        avg_depth_dict[word] = depth_sum/freq

    return avg_depth_dict


def is_valid_word(word):
    for letter in word:
        if not ((0x30 <= ord(letter) and ord(letter) <= 0x39) or \
        (0x61 <= ord(letter) and ord(letter) <= 0x7a)):
            return False

    if word in stopwords.words('English'):
        return False

    return True


def tf(document):
    word_count_dict = dict()
    for sentence in document:
        for w in sentence:
            if is_valid_word(w):
                if w in word_count_dict.keys():
                    word_count_dict[w] += 1
                else:
                    word_count_dict[w] = 1

    highest_frequency = 0
    for word, freq in word_count_dict.items():
        if freq > highest_frequency:
            highest_frequency = freq
    
    tf_dict = dict()

    for word,freq in word_count_dict.items():
        tf_dict[word] = 0.5 + 0.5*(freq/highest_frequency)

    return tf_dict

def get_word_count_dict(data):
    word_count_dict = dict()
    for i in range(0, num_documents):
        document = load_document(data["content"][i])
        for sentence in document:
            for w in sentence:
                if w in word_count_dict.keys():
                    word_count_dict[w] += 1
                else:
                    word_count_dict[w] = 1
    
    return word_count_dict

def contains_word(document, word):
    for sentence in document:
        if word in sentence:
            return True
    return False

def idf(data):
    word_count_dict = get_word_count_dict(data)
    num_document_dict = dict()
    for i in range(0, num_documents):
        document = load_document(data["content"][i])
        for word in word_count_dict.keys():
            if contains_word(document, word):
                if word in num_document_dict.keys():
                    num_document_dict[word] += 1
                else:
                    num_document_dict[word] = 1

    idf_dict = dict()
    for word, freq in num_document_dict.items():
        idf_dict[word] = math.log(num_documents/freq)
    
    return idf_dict

if __name__ == "__main__":

    data = pd.read_csv('question_corpus.csv')
    data = data[:300]

    idf_dict = idf(data)

    noun_symbols = ['NN', 'NNS', 'NNP', 'NNPS']

    fp = open("output.csv", 'w')
    csvwriter = csv.writer(fp)

    for i in range(0, num_documents):
        word_importance = list()
        document = load_document(data["content"][i])
        word_dict = build_word_dict(document)
        grammar = build_grammar(word_dict)
        avg_depth_dict = get_avg_depth_dict(document, grammar)

        tf_dict = tf(document)

        for sentence in document:
            tagged_sentence = nltk.pos_tag(sentence)
            for word, tag in tagged_sentence:
                if tag  in noun_symbols and word not in [item[0] for item in word_importance]:
                    try:
                        word_importance.append((word, tf_dict[word]*idf_dict[word]*avg_depth_dict[word]))
                    except:
                        print("eleminate")
        
        word_importance.sort(key = lambda item : item[1], reverse = True)

        tags = word_importance[:5]
        csvwriter.writerow(tags)

    fp.close()




        
        
    #output  = load_document(filename)