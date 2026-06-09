import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

# 전처리된 영화 리뷰 데이터를 읽어온다.
# TF-IDF는 문장을 숫자 벡터로 바꾸는 작업이므로,
# 보통 job01에서 만든 것처럼 정리된 리뷰 데이터를 사용하는 것이 좋다.
df_reviews = pd.read_csv('datasets/reviews_2017_2022.csv')

# 데이터의 컬럼 이름, 행 개수, 결측치 여부 등을 확인한다.
# TF-IDF를 적용할 리뷰 컬럼이 존재하는지 확인하는 용도이다.
df_reviews.info()

# TfidfVectorizer는 텍스트 문장을 TF-IDF 숫자 행렬로 변환하는 도구이다.
# TF-IDF는 단어가 한 문서 안에서 얼마나 자주 등장하는지(TF)와
# 전체 문서들 중 얼마나 희귀한 단어인지(IDF)를 함께 반영한다.
#
# sublinear_tf=True는 단어 빈도를 그대로 쓰지 않고 로그 스케일로 줄여준다.
# 예를 들어 어떤 단어가 한 리뷰에 100번 나왔다고 해서
# 1번 나온 단어보다 100배 중요하다고 보지 않도록 완화하는 옵션이다.
Tfidf = TfidfVectorizer(sublinear_tf=True)

# fit_transform은 두 가지 일을 한 번에 수행한다.
# 1. fit: 전체 리뷰를 훑으면서 단어 사전을 만든다.
#    예: {'재밌다': 0, '배우': 1, '스토리': 2, ...}
# 2. transform: 각 리뷰를 단어별 TF-IDF 점수 벡터로 변환한다.
#
# 결과인 Tfidf_matrix는 scipy sparse matrix 형태이다.
# 대부분의 단어는 각 리뷰에 등장하지 않기 때문에 0이 많고,
# 이런 데이터를 효율적으로 저장하기 위해 희소 행렬을 사용한다.
Tfidf_matrix = Tfidf.fit_transform(df_reviews.reviews)

# TF-IDF 행렬의 크기를 출력한다.
# shape 결과는 (리뷰 개수, 단어 개수) 형태이다.
# 예: (3174, 50000)이면 리뷰 3174개를 50000개 단어 기준의 숫자 벡터로 바꾼 것이다.
print(Tfidf_matrix.shape)

with open('./models/tfidf.pkl','wb') as f:
    pickle.dump(Tfidf, f)
mmwrite('./models/Tfidf_movie_review.mtx',Tfidf_matrix)
