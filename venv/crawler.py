from bs4 import BeautifulSoup
import requests
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

base_url = "https://api.stackexchange.com/2.2/questions/61522715"

params = {
    "site": "stackoverflow",
    'key': 'bekvFShStjiqRY6zNwJNHA(('
}

page = requests.get(base_url, params=params).json()

meta_data = []

items = page['items'][0]
tags = items['tags']
answer_count = items['answer_count']
last_edit_date = items['last_edit_date']
question_id = items['question_id']
link = items['link']
title = items['title']

html = requests.get(link)
soup = BeautifulSoup(html.text, "html.parser")
content = soup.findAll(class_='post-text')

meta_data.append((question_id, title, link, content, tags, last_edit_date, answer_count))
df = pd.DataFrame(meta_data, columns=['id', 'title', 'link', 'content', 'tags', 'last_edit', 'answer_count'])
print(df.head())
