import pandas as pd
from konlpy.tag import Okt
import re

# 원본 영화 리뷰 데이터를 읽어온다.
# 아래 전처리의 핵심 대상은 reviews 컬럼이다.
df = pd.read_csv('datasets/reviews_2017_2022.csv')

# 원본 데이터의 컬럼 이름, 결측치 여부, 데이터 개수 등을 확인한다.
df.info()

# 분석에 큰 의미가 없거나 너무 자주 등장하는 단어 목록을 읽어온다.
df_stopwords = pd.read_csv('datasets/stopwords.csv')

# stopwords.csv의 stopword 컬럼을 파이썬 리스트로 변환한다.
stopwords = df_stopwords['stopword'].tolist()
stopwords = stopwords + ['가다', '감독', '연출', '연기', '배우', '하다', '모르다', '보여주다', '주연', '많다'
                         , '좋다']


# Okt는 KoNLPy에서 제공하는 한국어 형태소 분석기이다.
# 문장을 명사, 동사, 형용사 같은 품사 단위로 쪼개기 위해 사용한다.
okt = Okt()

# 첫 번째 영화 제목과 리뷰를 출력해서 데이터가 정상적으로 읽혔는지 확인한다.
print(df.titles[0])
print(df.reviews[0])

# 첫 번째 리뷰의 형태소 분석 결과를 확인하고 싶을 때 사용하는 테스트 코드이다.
# tokened_review = okt.pos(df.reviews[0])
# print(tokened_review)

# 전처리가 완료된 리뷰 문장들을 저장할 리스트이다.
cleaned_sentences = []

# 원본 데이터의 reviews 컬럼을 한 줄씩 가져와서 전처리한다.
for review in df.reviews:
    # 한글이 아닌 문자는 모두 공백으로 바꾼다.
    # 숫자, 영어, 특수문자, 이모티콘 등을 제거해서 한국어 단어만 남긴다.
    review = re.sub('[^가-힣]', ' ', review)

    # 형태소 분석을 수행한다.
    # stem=True는 "좋았다", "좋은" 같은 활용형을 기본형에 가깝게 정리한다.
    tokened_review = okt.pos(review, stem=True)

    # 형태소 분석 결과는 [('단어', '품사'), ...] 형태이므로
    # 단어와 품사를 다루기 쉽게 DataFrame으로 변환한다.
    df_token = pd.DataFrame(tokened_review, columns=['word', 'class'])

    # 리뷰의 의미를 파악하는 데 비교적 중요한 품사만 남긴다.
    # Noun: 명사, Verb: 동사, Adjective: 형용사
    # 조사, 어미 등은 분석에서 잡음이 되기 쉬워 제거한다.
    df_token = df_token[(df_token['class'] == 'Noun')
                        | (df_token['class'] == 'Verb')
                        | (df_token['class'] == 'Adjective')]

    # 현재 리뷰에서 최종적으로 남길 단어들을 저장한다.
    words = []
    for word in df_token['word']:
        # 한 글자 단어는 의미가 약하거나 잡음일 가능성이 높아 제거한다.
        if len(word) > 1:
            # 불용어 목록에 포함되지 않은 단어만 남긴다.
            if word not in stopwords:
                words.append(word)

    # 남은 단어들을 공백으로 이어 붙여 하나의 전처리된 리뷰 문장으로 만든다.
    cleaned_sentence = ' '.join(words)

    # 완성된 전처리 리뷰를 전체 결과 리스트에 추가한다.
    cleaned_sentences.append(cleaned_sentence)

# 원본 reviews 컬럼을 전처리된 리뷰 문장으로 교체한다.
df.reviews = cleaned_sentences

# 결측치가 있는 행을 제거한다.
# 단, 빈 문자열은 NaN이 아니므로 이 코드만으로 완전히 빈 리뷰가 제거되지는 않는다.
df.dropna(inplace=True)

# 전처리 후 데이터 상태를 다시 확인한다.
df.info()

# 전처리된 데이터를 새 CSV 파일로 저장한다.
# 이후 감성 분석, 키워드 분석, 머신러닝 학습 등에 이 파일을 사용할 수 있다.
df.to_csv('./datasets/reviews_2017_2022_test.csv', index=False)
