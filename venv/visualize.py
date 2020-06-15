import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import ast

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', None)


def collapse_pairs(pairs):
    try:
        pairs = ast.literal_eval(pairs)
    except ValueError:
        pairs = set()
    return list({pair[1] for pair in pairs})


df_word_eval = pd.read_csv('eval-output_word.csv')
df_sent_eval = pd.read_csv('eval-output_sent.csv')
df_word_output = pd.read_csv('output_word.csv')
df_sent_output = pd.read_csv('output_sent.csv')
df_corpus = pd.read_csv('question_corpus.csv')
final_df = pd.DataFrame()
final_df['text'] = df_corpus['content']
final_df['real_tags'] = df_corpus.apply(lambda r: ast.literal_eval(r['tags']), axis=1)
final_df['predicted_tags_from_words'] = df_word_output.apply(lambda r: ast.literal_eval(r['tags']), axis=1)
final_df['predicted_tags_from_sents'] = df_sent_output.apply(lambda r: ast.literal_eval(r['tags']), axis=1)
final_df['ratio_correct_from_words'] = df_word_eval['ratio-correct']
final_df['ratio_correct_from_sents'] = df_sent_eval['ratio-correct']
final_df['matched_tags_from_words'] = df_word_eval['pairs-match']
final_df['matched_tags_from_sents'] = df_sent_eval['pairs-match']
final_df['matched_tags_from_words'] = final_df.apply(lambda r: collapse_pairs(r['matched_tags_from_words']), axis=1)
final_df['matched_tags_from_sents'] = final_df.apply(lambda r: collapse_pairs(r['matched_tags_from_sents']), axis=1)
final_df = final_df[:300]
print(final_df.head(10))


def draw_histogram(data, filename):
    fig = plt.figure()
    sns.distplot(data, kde=False)
    fig.savefig(filename)
    plt.show()


draw_histogram(final_df.ratio_correct_from_words, 'imgs/histogram_ratio_correct_from_words.png')
draw_histogram(final_df.ratio_correct_from_sents, 'imgs/histogram_ratio_correct_from_sents.png')


def draw_wordcloud(column):
    predicted_tags_from_words = final_df[column].tolist()
    # predicted_tags_from_words = list(map(lambda p: ast.literal_eval(p), predicted_tags_from_words))
    predicted_tags_from_words = [item.strip() for sublist in predicted_tags_from_words for item in sublist]
    predicted_tags_from_words = ' '.join(predicted_tags_from_words)
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(predicted_tags_from_words)
    wordcloud.to_file('imgs/wordcloud_{}.png'.format(column))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


draw_wordcloud('predicted_tags_from_words')
draw_wordcloud('predicted_tags_from_sents')
draw_wordcloud('real_tags')
draw_wordcloud('matched_tags_from_words')
draw_wordcloud('matched_tags_from_sents')
