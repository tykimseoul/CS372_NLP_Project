from bs4 import BeautifulSoup
import requests
import pandas as pd

FILE_CORPUS = 'question_corpus.csv'


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

page_size = 20
base_url = "https://api.stackexchange.com/2.2/questions"

params = {
    "site": "stackoverflow",
    'key': 'bekvFShStjiqRY6zNwJNHA((',
    'sort': 'activity',
    'order': 'desc',
    'page': 1,
    'pagesize': page_size
}


def parse_question(question):
    question_id = question['question_id']
    tags = question['tags']
    answer_count = question['answer_count']
    last_activity_date = question['last_activity_date']
    link = question['link']
    title = question['title']
    
    html = requests.get(link)
    soup = BeautifulSoup(html.text, "html.parser")
    content = soup.findAll(class_='post-text')
    
    return question_id, title, link, content, tags, last_activity_date, answer_count


meta_data = []

for i in range(1, int(10000 / page_size) + 1):
    print('crawling page {}'.format(i))
    params['page'] = i
    page = requests.get(base_url, params=params).json()
    has_more = page['has_more']
    quota_remaining = page['quota_remaining']
    print('quota remaining: {}'.format(quota_remaining))
    if not has_more or quota_remaining == 0:
        print('crawling terminated')
    items = page['items']
    meta_data.extend(map(lambda q: parse_question(q), items))

df = pd.DataFrame(meta_data, columns=['id', 'title', 'link', 'content', 'tags', 'last_activity', 'answer_count'])
df.drop_duplicates(subset='id', keep='first', inplace=True)
df.to_csv(FILE_CORPUS, index=False, header=True)
print(df.head(10))
print(df.describe())
