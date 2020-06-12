import nltk
import pandas as pd
import re
import csv
import math
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
stm = LancasterStemmer()
from collections import defaultdict
import requests
import ast

NUM_DOCUMENTS = 50
FILE_CORPUS = 'question_corpus.csv'
FILES_OUTPUT = ['output_word.csv']
#FILES_OUTPUT = ['output_word.csv', 'output_sent.csv']
FILE_OUTPUT_WORD = 'output_word.csv'
FILE_OUTPUT_SENT = 'output_sent.csv'
#FILE_EVAL = 'eval_result.csv'

STRICT = False
INCLUSIVE = True
SEARCH_SUGGESTION = True
NUM_TAGS_POP = 1000
FILE_HEAD = 'eval-'

# Evaluates:
# - how well the models correctly predicted the actual tags 
# - whether the predicted tags that are not in the actual tags,
#       are used by other posts on Stackoverflow (and therefore can be considered as a valid suggestion)
#       enable/disable with SEARCH_SUGGESTION

# Our current models all return single-word tags only
# however on stackoverflow there are tags that consists of multiple words
# .. which, conventionally, look like this: foo-bar-baz

# Methods of tag matching
# - strict: only marks exact matches
# - lenient: special characters and numbers are discarded when matching,
#       also pairs are considered 'matching' if they are different pos forms of the same base word
#       e.g.) 'foo4.1' and 'foo', 'barred' and 'barring'

# - inclusive strict: also marks predicted tags that strictly matches a word in a multi-word actual tag
#       e.g.) 'foo-bar' and 'bar'
# - inclusive lenient: marks predicted tags that leniently matches a word in a multi-word actual tag

# identifies whether word1(str) and word2(str) match strictly/leniently, according to strict(bool)
def match_word(word1, word2, strict):
    if (strict): return (word1==word2)
    else:
        word1_alpha = ''.join(e for e in word1 if e.isalpha())
        word2_alpha = ''.join(e for e in word2 if e.isalpha())
        if (word1_alpha == word2_alpha): return True
        if (len(wn.synsets(word1_alpha))>0 and len(wn.synsets(word2_alpha))>0):
            if(stm.stem(word1_alpha) == stm.stem(word2_alpha)):
                return True
        return False

# identifies whether word1(str) matches words2(list(str))
# if inclusive == True, checks whether word1 matches an element of words2
# if strict == False, lenient matches are allowed
def match_words(word1, words2, inclusive, strict):
    if (not inclusive): return match_word(word1, '-'.join(words2), strict)
    else:
        for word2 in words2:
            if(match_word(word1, word2, strict)): return True
    return False


# Collects the NUM_TAGS_POP most popular tags on stackoverflow and their usage-counts 

page_size = 20
base_url = "https://api.stackexchange.com/2.2/tags"
params = {
    "site": "stackoverflow",
    'key': 'bekvFShStjiqRY6zNwJNHA((',
    'sort': 'popular',
    'order': 'desc',
    'page': 1,
    'pagesize': page_size
}

tags_pop = list()

for i in range(1, int(NUM_TAGS_POP / page_size) + 1):
    params['page'] = i
    page = requests.get(base_url, params=params).json()
    has_more = page['has_more']
    quota_remaining = page['quota_remaining']
    items = page['items']
    tags_pop.extend(map(lambda item: (item['name'], item['count']), items))



data_corpus = pd.read_csv(FILE_CORPUS)[:NUM_DOCUMENTS]

# Each entry of tags_real is a list
# ..whose entries are, in turn, lists of the tags split by '-'
# e.g.) if the tags were [foo, bar-baz], the corresponding entry in tags_real would be [[foo],[bar,baz]]  
tags_real = []
for strtags in data_corpus['tags']:
    tags_real.append([tag.split('-') for tag in ast.literal_eval(strtags)])

# Store evaluation results of file_output for i-th document in result_eval[i][file_output]
result_eval = defaultdict(defaultdict)

for file_output in FILES_OUTPUT:
    data_output = pd.read_csv(file_output)
    strtags_output = data_output['tags']
    
    for i in range(NUM_DOCUMENTS):
        
        result_eval[i][file_output] = dict()
        
        tags_correct = set() # predicted tags that match one of the actual tags
        tags_missed = list() # predicted tags that do not match any of the actual tags
        tags_leftout = set() # actual tags that match with none of the predicted tags 
        
        tags_output = ast.literal_eval(strtags_output[i])
        
        for tag_words in tags_real[i]:
            found_match = False
            for tag_output in tags_output:
                if(match_words(tag_output, tag_words, INCLUSIVE, STRICT)):
                    found_match = True
                    tags_correct.add(tag_output)
            if (found_match):
                tags_correct.add(tag_output)
            else:
                tags_leftout.add('-'.join(tag_words))
        
        tags_missed = [tag_output for tag_output in tags_output if tag_output not in tags_correct]
        result_eval[i][file_output]['ratio-correct'] = (len(tags_correct)/len(tags_output)) if (len(tags_output)>0) else 0
        result_eval[i][file_output]['ratio-match'] = (len([tag_real for tag_real in tags_real[i] if '-'.join(tag_real) not in tags_leftout])/len(tags_real[i])) if (len(tags_real[i])>0) else 0
        result_eval[i][file_output]['tags-missed'] = tags_missed
        result_eval[i][file_output]['tags-leftout'] = tags_leftout
        
        result_eval[i][file_output]['valid-suggestions'] = list()
        for tag_missed in tags_missed:
            for (tag_pop, count) in tags_pop:
                if(match_words(tag_missed, tag_words, True, False)):
                    result_eval[i][file_output]['valid-suggestions'].append((tag_missed, tag_pop, count))
    
    result = list()
    for i in range(NUM_DOCUMENTS):
        result.append( tuple(result_eval[i][file_output].values()))
    
    df = pd.DataFrame(result, columns=['tags-missed','tags-leftout','valid-suggestions'])
    #df = pd.DataFrame(result, columns=['ratio-correct','ratio-match','tags-missed','tags-leftout','valid-suggestions'])
    df.to_csv(FILE_HEAD+file_output, index=False, header=True)
