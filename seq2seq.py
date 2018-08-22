# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
# 영어 단어를 한국어 단어로 번역하는 프로그램을 만들어봅니다.
import tensorflow as tf
import numpy as np


seq_data = [
    ["당뇨병의 정의",
     "당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다. "
     "포도당은 우리가 먹는 음식물 중 탄수화물의 기본 구성성분입니다. "
     "탄수화물은 위장에서 소화효소에 의해 포도당으로 변한 다음 혈액으로 흡수됩니다. "
     "흡수된 포도당이 우리 몸의 세포들에서 이용되기 위해서는 인슐린이라는 호르몬이 반드시 필요합니다. "
     "인슐린은 췌장 랑게르한스섬에서 분비되어 식사 후 올라간 혈당을 낮추는 기능을 합니다. "
     "만약 여러 가지 이유로 인하여 인슐린이 모자라거나 성능이 떨어지게 되면, 체내에 흡수된 포도당은 이용되지 못하고 혈액 속에 쌓여 소변으로 넘쳐 나오게 되며, 이런 병적인 상태를 '당뇨병' 이라고 부르고 있습니다. "
     "우리나라도 최근 들어 사회 경제적인 발전으로 과식, 운동부족, 스트레스 증가 등으로 인하여 당뇨병 인구가 늘고 있습니다. "
     "2010년 통계를 보면 우리나라의 전체 인구 중 350만명 정도가 당뇨병환자인 것으로 추정되고 있으나, 이중의 반 이상은 아직 자신이 당뇨병환자임을 모르고 지냅니다. "],
    ["당뇨병의 원인", "유전적 요인, 환경적 요인 "],
    ["유전적 요인",
     "당뇨병의 발병 원인은 아직 정확하게 규명이 되어있지 않습니다. "
     "현재까지 밝혀진 바에 의하면 유전적 요인이 가장 가능성이 큽니다. "
     "만약, 부모가 모두 당뇨병인 경우 자녀가 당뇨병이 생길 가능성은 30% 정도이고, 한 사람만 당뇨병인 경우는 15% 정도입니다. "
     "하지만 유전적 요인을 가지고 있다고 해서 전부 당뇨병환자가 되는 것은 아니며, 유전적인 요인을 가진 사람에게 여러 가지 환경적 요인이 함께 작용하여 당뇨병이 생기게 됩니다. "],
    ["환경적 요인",
     "비 만 - 뚱뚱하면 일단 당뇨병을 의심하라는 말이 있듯이 비만은 당뇨병과 밀접한 관련이 있습니다. "
     "계속된 비만은 몸 안의 인슐린 요구량을 증가시키고, 그 결과로 췌장의 인슐린 분비기능을 점점 떨어뜨려 당뇨병이 생깁니다. "
     "또한 비만은 고혈압이나 심장병의 원인이 되기도 합니다. "
     "연령 - 당뇨병은 중년 이후에 많이 발생하며 연령이 높아질수록 발병률도 높아집니다. "
     "식생활 - 과식은 비만의 원인이 되고, 당뇨병을 유발하므로 탄수화물(설탕포함)과 지방의 과다한 섭취는 피해야 합니다. "
     "운동부족 - 운동부족은 고혈압, 동맥경화 등 성인병의 원인이 됩니다. "
     "운동부족은 비만을 초래하고, 근육을 약화시키며, 저항력을 저하시킵니다. "
     "스트레스 - 우리 몸에 오래 축적된 스트레스는 부신피질호르몬의 분비를 증가시키고, 저항력을 떨어뜨려 질병을 유발합니다."
     "성별 - 일반적으로 여성이 남성보다 발병률이 높습니다.그 이유는 임신이라는 호르몬 환경의 변화 때문입니다. "
     "호르몬 분비 당뇨병과 직접 관련이 있는 인슐린과 글루카곤 호르몬에 이상이 생기면 즉각적으로 당뇨병이 유발되며, 뇌하수체나 갑상선, 부신호르몬과 같은 간접적인 관련인자도 당뇨병을 일으킬 수 있습니다. "
     "감염증 감염증에 걸리면 신체의 저항력이 떨어지고, 당대사도 나빠지게 되어 당뇨병이 발생하기 쉽습니다.특히 췌장염, 간염,담낭염 등은 당뇨병을 일으킬 가능성이 크므로 신속하게 치료해야 합니다. "
     "약물복용 다음과 같은 약물을 장기간 사용하는 경우에는 당뇨병 소질을 갖고 있는 사람에게 영향을 끼칠 수 있습니다. "
     "호르몬 분비 - 당뇨병과 직접 관련이 있는 인슐린과 글루카곤 호르몬에 이상이 생기면 즉각적으로 당뇨병이 유발되며, 뇌하수체나 갑상선, 부신호르몬과 같은 간접적인 관련인자도 당뇨병을 일으킬수 있습니다. "
     "감염증 - 감염증에 걸리면 신체의 저항력이 떨어지고, 당대사도 나빠지게 되어 당뇨병이 발생하기 쉽습니다.특히 췌장염, 간염, 담낭염 등은 당뇨병을 일으킬 가능성이 크므로 신속하게 치료해야 합니다. "
     "약물복용 - 다음과 같은 약물을 장기간 사용하는 경우에는 당뇨병 소질을 갖고 있는 사람에게 영향을 끼칠 수 있습니다. "
     "① 신경통, 류마티즘, 천식, 알레르기성 질환 등에 사용하는 부신피질 호르몬제 "
     "② 혈압을 내리고 이뇨작용을 하는 강압 이뇨제 "
     "③ 경구용 피임약 "
     "④ 소염 진통제 "
     "⑤ 갑상선 호르몬제 외과적 수술 위절제 수술 후 당대사에 이상이 생기는 경우가 있습니다. "
     "따라서 위절제 수술을 받은 사람이면서, 당뇨병 소질을 갖고 있는 경우는 혈당의 변동을 주의  깊게 살펴야 합니다. "
     "외과적 수술 - 위절제 수술 후 당대사에 이상이 생기는 경우가 있습니다. "
     "따라서 위절제 수술을 받은 사람이면서, 당뇨병 소질을 갖고 있는 경우는 혈당의 변동을 주의 깊게 살펴야 합니다. "],
    ["당뇨병의 증상",
     "혈당이 높아지면 소변으로 당이 빠져나가게 되는데, 이때 포도당이 다량의 물을 끌고 나가기 때문에 소변을 많이 보게 됩니다. "
     "따라서 몸 안의 수분이 모자라 갈증이 심하며 물을 많이 마시게 됩니다. 또한, 우리가 섭취한 음식물이 소변으로 빠져나가 에너지로 이용되지 못하므로 공복감은 심해지고 점점 더 먹으려 합니다. "
     "당뇨병의 3대 증상은 다음(多飮), 다식(多食), 다뇨(多尿)이지만 이외에도 여러 증상이 있습니다. 당뇨병은 특별한 증상이 없을 수도 있어, 자신이 당뇨병인지 모르고 지내다가 뒤늦게 진단받는 경우도 있습니다. "],
    ["당뇨병의 분류", "제 1형 당뇨병, 제 2형 당뇨병, 기타 형태의 당뇨병, 임신성 당뇨병 "],
    ["제 1형 당뇨병",
     "당뇨병 우리나라 당뇨병의 2% 미만을 차지하며 주로 소아에서 발생하나, 성인에서도 나타날 수 있습니다. "
     "급성 발병을 하며 심한 다음, 다뇨, 체중감소 등과 같은 증상들이 나타나고, 인슐린의 절대적인 결핍으로 인하여 케톤산증이 일어납니다. "
     "고혈당의 조절 및 케톤산증에 의한 사망을 방지하기 위해 인슐린치료가 반드시 필요합니다. "],
    ["제 2형 당뇨병","당뇨병 한국인 당뇨병의 대부분을 차지하며 체중정도에 따라서 비만형과 비비만형으로 나눕니다. "],
    ["기타 형태의 당뇨병", "췌장질환, 내분비질환, 특정한 약물, 화학물질, 인슐린 혹은 인슐린 수용체 이상, 유전적 증후군에 의해 2차적으로 당뇨병이 유발되는 경우가 있습니다. "],
    ["임신성 당뇨병","임신성 당뇨병이란 임신 중 처음 발견되었거나 임신의 시작과 동시에 생긴 당조절 이상을 말하며 임신 전 진단된 당뇨병과는 구분됩니다. "],
    ["당뇨병의 진단","당뇨병의 증상은 다양하며 때로는 전혀 증상이 없는 경우도 있습니다. "],
]

def load_data(sequence):
    question = []
    answer = []
    for sentence in sequence:
        question.append(''.join(sentence[0]).split())
        answer.append(''.join(sentence[1]).split())

    print("question: ", question)
    print("answer: ", answer)
    return question, answer

def make_dic(sequence):
    words = []
    words.append('<PAD>')
    words.append('<S>')
    words.append('<E>')
    words.append('<UNK>')
    for seq in sequence:
        for word in seq:
            words.append(word)
    words = sorted(list(set(words)))
    # word_to_ix, ix_to_word 생성
    idx2word = {idx: word for idx, word in enumerate(words)}
    word2idx = {word: idx for idx, word in enumerate(words)}
    print('컨텐츠 갯수 : %s, 단어 갯수 : %s'% (len(sequence), len(idx2word)))
    print("word2idx: ", word2idx)
    print("idx2word:", idx2word)
    return word2idx, idx2word


def check_length(sequence):
    max_size = 0
    for seq in sequence:
        if max_size < len(seq):
            max_size = len(seq)
    print("max_size:", max_size)
    return max_size

def check_length(sequence):
    max_size = 0
    for seq in sequence:
        if max_size < len(seq):
            max_size = len(seq)
    print("max_size:", max_size)
    return max_size


# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
question, answer = load_data(seq_data)
word2idx, idx2word = make_dic(question + answer)
vocab_size = len(word2idx)

def make_batch(encoder, decoder):
    input_batch = []
    output_batch = []
    target_batch = []

    for enc, dec in zip(encoder, decoder):
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
        tmp_encoder_input = [word2idx[n] for n in enc]
        encoder_padd_size = max(vocab_size - len(tmp_encoder_input), 0)
        encoder_padd = [word2idx['<PAD>']] * encoder_padd_size
        input_batch.append(np.eye(vocab_size)[tmp_encoder_input + encoder_padd])
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        tmp_decoder_input = [word2idx[c] for c in dec]
        decoder_padd_size = vocab_size - len(tmp_decoder_input) - 1
        decoder_padd = [word2idx['<PAD>']] * decoder_padd_size
        output_batch.append(np.eye(vocab_size)[[word2idx['<S>']] + tmp_decoder_input + decoder_padd])
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target_batch.append(tmp_decoder_input + [word2idx['<E>']] + decoder_padd)
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
    return input_batch, output_batch, target_batch


#########
# 옵션 설정
######
learning_rate = 0.01
n_hidden = 128
total_epoch = 500
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.


#########
# 신경망 모델 구성
######
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, vocab_size])
dec_input = tf.placeholder(tf.float32, [None, None, vocab_size])
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])


# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)


model = tf.layers.dense(outputs, vocab_size, activation=None)


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(question, answer)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')


#########
# 번역 테스트
######
# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    empty_dec = '<PAD> ' * len(word)
    seq_data = [''.join(word).split(), ''.join(empty_dec).split()]
    print(seq_data[0])
    print(seq_data[1])
    input_batch, output_batch, target_batch = make_batch([seq_data[0]], [seq_data[1]])

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [idx2word[i] for i in result[0]]
    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
#    end = decoded.index('<E>')
    translated = ''.join(decoded[:-1])

    return translated


print('\n=== 번역 테스트 ===')

print('당뇨병의 정의 ->', translate('당뇨병의 정의'))
