import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle


def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[0]))
    simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
    simScore = simScore[1:11]
    movieIdx = [i[0] for i in simScore]
    recmovieList = df_reviews.loc[movieIdx, 'titles']
    return recmovieList


df_reviews = pd.read_csv('./datasets/reviews_2017_2022.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()

with open('./models/tfidf.pkl', 'rb') as f:
    Tfidf = pickle.load(f)

ref_idx = 16
print('title:', df_reviews.iloc[ref_idx]['titles'])

cosine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix)
print(cosine_sim[0])
print(len(cosine_sim[0]))

recommendation = getRecommendation(cosine_sim)
print(recommendation)
