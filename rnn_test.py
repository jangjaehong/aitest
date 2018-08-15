import tensorflow as tf
import numpy as np
import func

seq_data = [
    ["당뇨병의 정의", "당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다."],
    ["당뇨병의 원인", "현재까지 밝혀진 바에 의하면 유전적 요인이 가장 가능성이 큽니다."]
]

question, answer = func.load_data(seq_data)
word2idx, idx2word = func.make_dic(question + answer, min_length=0, max_length=3, jamo_delete=True)
dic_len = len(word2idx)

encoder_size = func.check_length(question)
decoder_size = func.check_length(answer)
encoder_seq_len = func.sequence_length(question)
decoder_seq_len = func.sequence_length(answer)

learning_rate = 0.01
n_hidden = 128
total_epoch = 1000
n_class = n_input = dic_len
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
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

encoder_inputs, decoder_inputs, target_inputs = func.make_batch(
    rawencoder=question,
    rawdecoer=answer,
    word2idx=word2idx,
    encoder_size=encoder_size,
    decoder_size=decoder_size,
    dic_len=dic_len)

for epoch in range(total_epoch):
    _, loss, results = sess.run([optimizer, cost, outputs],
                       feed_dict={enc_input: encoder_inputs,
                                  dec_input: decoder_inputs,
                                  targets: target_inputs})
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))


#########
# 번역 테스트
######
# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    seq_data = [[word, '<PAD> ' * decoder_size]]
    print(seq_data)
    question, answer = func.load_data(seq_data)
    encoder_inputs, decoder_inputs, target_inputs = func.make_batch(
        rawencoder=question,
        rawdecoer=answer,
        word2idx=word2idx,
        encoder_size=encoder_size,
        decoder_size=decoder_size,
        dic_len=dic_len)
    print(encoder_inputs)
    print(decoder_inputs)
    print(target_inputs)
    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: encoder_inputs,
                                 dec_input: decoder_inputs,
                                 targets: target_inputs})
    print(result)
    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [idx2word[i] for i in result[0]]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    end = decoded.index('<E>')
    translated = ' '.join(decoded[:end])

    return translated

print('\n=== 번역 테스트 ===')

print('당뇨병의 정의 ->', translate('당뇨병의 정의'))
