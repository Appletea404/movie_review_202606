import pandas as pd
from wordcloud import WordCloud
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font',family= 'NanumBarunGothic')

df = pd.read_csv('datasets/reviews_2017_2022.csv')

movie_index1 = 16

words = df.iloc[movie_index1,1].split()
print(df.iloc[movie_index1,0])

movie_index2 = 308

words1 = df.iloc[movie_index1,1].split()
print(df.iloc[movie_index1,0])


words2 = df.iloc[movie_index2,1].split()
print(df.iloc[movie_index2,0])


# words = df.iloc[16,1].split()
# print(df.iloc[16,0])

worddict1 = collections.Counter(words1)
worddict1 = dict(worddict1)

worddict2 = collections.Counter(words2)
worddict2 = dict(worddict2)

print(worddict1)
print(worddict2)

wordcloud1 = WordCloud(font_path=font_path, background_color='white').generate_from_frequencies(worddict1)
wordcloud2 = WordCloud(font_path=font_path, background_color='white').generate_from_frequencies(worddict2)




plt.subplot(1, 2, 1)
plt.imshow(wordcloud1)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(wordcloud2)
plt.axis("off")

plt.show()
