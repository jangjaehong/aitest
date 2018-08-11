from tensorflow.contrib import rnn
import numpy as np
import tensorflow as tf
from konlpy.tag import Okt

chat_log = {"당뇨병의 정의": "당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다. 포도당은 우리가 먹는 음식물 중 탄수화물의 기본 구성성분입니다.",
            "당뇨병의 원인": "유전적 요인 환경적 요인",
            "당뇨병의 증상": "혈당이 높아지면 소변으로 당이 빠져나가게 되는데, 이때 포도당이 다량의 물을 끌고 나가기 때문에 소변을 많이 보게 됩니다. 따라서 몸 안의 수분이 모자라 갈증이 심하며 물을 많이 마시게 됩니다. 또한, 우리가 섭취한 음식물이 소변으로 빠져나가 에너지로 이용되지 못하므로 공복감은 심해지고 점점 더 먹으려 합니다.",
            "당뇨병의 분류": "제 1형 당뇨병 제 2형 당뇨병 기타 형태의 당뇨병 임신성 당뇨병",
            "당뇨병의 진단": "당뇨병의 증상은 다양하며 때로는 전혀 증상이 없는 경우도 있습니다.",
            "당뇨병이란?": "인슐린은 췌장 랑게르한스섬에서 분비되어 식사 후 올라간 혈당을 낮추는 기능을 합니다. "}


def parse(sentence, depth=1):
    twitter = Okt()
    result = []
    if depth == 1:
        return twitter.morphs(sentence)
    elif depth == 2:
        for char in sentence:
            result = result + twitter.morphs(char)
        return list(set(result))


####################################
# 채팅을 위한 기본적인 질의문 로드 #
####################################
# dict-key:질의문, value:답변문, 각각 리스트 형식으로 생성
question_context = list(chat_log.keys())
answer_context = list(chat_log.values())

#################################
# 데이터를 학습 시키기위한 작업 #
#################################
# 질의문과 답변문을 하나의 리스트로 합친다, 그리고 S:stratr, E:end, P,를
# 추가하여 추후 입력데이터를 생성시 구분자로써 사용한다.
qa_context = question_context + answer_context + ["S", "E", "P"]
# konlpy에서 mecab을 사용하여 sequence를 분석한다.
qa_arr = parse(qa_context, depth=2)
# 분석 결과 분류된 단어에 번호를 부여한다.
qa_dic = {n: i for i, n in enumerate(qa_arr)}
# 단어의 총 갯수
dic_len = len(qa_dic)

# print("answer_context:", answer_context)
# print("qa_context: ", qa_context)
# print("qa_arr: ", qa_arr)
# print("qa_dic: ", qa_dic)
# print("dic_len: ", dic_len)

#######################################################
# 질의문에 따라 답변을 제시하기 위한 학습 데이터 생성 #
#######################################################
# 질의문과 답변문을 하나의 리스트로 작성
seq_data = []
for q, a in zip(qa_context, answer_context):
    seq_data.append([q, a])


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값, 입력문자열을 macab을 통해 단어로 분리 배열로 만든다.
        input = [qa_dic[n] for n in parse(seq[0], depth=1)]
        # 디코더 셀의 입력값, 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [qa_dic[n] for n in parse('S' + seq[1], depth=1)]
        # 학습을 위해 비교할 디코더 셀의 출력값, 끝나는 것을 알려주기 위해 마지막에 E를 붙인다.
        target = [qa_dic[n] for n in parse(seq[1] + 'E', depth=1)]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch
