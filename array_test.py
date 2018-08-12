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


def maximum_size(sequence):
    batch_size = 0
    for seq in sequence:
        seq_len = len(seq)
        if seq_len > batch_size:
            batch_size = seq_len

    return batch_size


def resize(sequence, black_value):
    # sequence의 max_length를 구한다.
    re_batch = []
    max_size = maximum_size(sequence)
    for seq in sequence:
        if len(seq) < max_size:
            for i in range(0, max_size - len(seq)):
                seq.append(black_value)
            #print(seq)
        re_batch.append(seq)
    return re_batch


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
# konlpy에서 mecab을 사용하여 sequence를 분석, array 형태로 생성
qa_list = parse(qa_context, depth=2)
# 분석 결과 분류된 단어에 번호를 부여한다.
qa_dic = {n: i for i, n in enumerate(qa_list)}
# 단어의 총 갯수
dic_len = len(qa_dic)

seq_data = []
for q, a in zip(question_context, answer_context):
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

    target_batch = resize(target_batch, qa_dic["P"])
    return input_batch, output_batch, target_batch


#########
# 옵션 설정
######
learning_rate = 0.01
n_hidden = 128
total_epoch = 100
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = dic_len


#########
# 신경망 모델 구성
######
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])


# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)


model = tf.layers.dense(outputs, n_class, activation=None)


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)
for i in target_batch:
    print(i)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

#     print('Epoch:', '%04d' % (epoch + 1),
#           'cost =', '{:.6f}'.format(loss))
#
# print('최적화 완료!')

#
# #########
# # 번역 테스트
# ######
# # 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
# def translate(word):
#     # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
#     # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
#     # ['word', 'PPPP']
#     seq_data = [word, 'P' * len(word)]
#
#     input_batch, output_batch, target_batch = make_batch([seq_data])
#
#     # 결과가 [batch size, time step, input] 으로 나오기 때문에,
#     # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
#     prediction = tf.argmax(model, 2)
#
#     result = sess.run(prediction,
#                       feed_dict={enc_input: input_batch,
#                                  dec_input: output_batch,
#                                  targets: target_batch})
#
#     # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
#     decoded = [char_arr[i] for i in result[0]]
#
#     # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
#     end = decoded.index('E')
#     translated = ''.join(decoded[:end])
#
#     return translated
#
#
# print('\n=== 번역 테스트 ===')
#
# print('word ->', translate('word'))
# print('wodr ->', translate('wodr'))
# print('love ->', translate('love'))
# print('loev ->', translate('loev'))
# print('abcd ->', translate('abcd'))
