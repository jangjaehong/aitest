from gensim.models import Word2Vec
from konlpy.tag import Okt
okt = Okt()
tokenized_contents = [okt.morphs("당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다")]
print(tokenized_contents)
embedding_model = Word2Vec(tokenized_contents, size=100, window=2, min_count=50, workers=4, iter=100, sg=1)
print(embedding_model.most_similar(positive=["당뇨병"], topn=100))