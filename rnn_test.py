import tensorflow as tf
import numpy as np
from konlpy.tag import Okt

okt = Okt()
question_data = "당뇨병의 정의", "당뇨병의 원인"
answer_data = "당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다.", "현재까지 밝혀진 바에 의하면 유전적 요인이 가장 가능성이 큽니다."
# in, target data
input_data = []
for seq in question_data:
    seq_morphs = okt.morphs(seq)
    input_data.append(seq_morphs)

target_data = []
for seq in answer_data:
    seq_morphs = okt.morphs(seq)
    target_data.append(seq_morphs)

 # dic 만들기
words = []
for seq in input_data + target_data:
    words = words + seq
words = list(set(words))
word2idx = {c: i for i, c in enumerate(words)}
idx2word = {i: c for i, c in enumerate(words)}
dic_len = len(word2idx)

print(input_data)
print(target_data)
# #parameter
# batch_size = 8
# sequence_length = 5
# # input_dim = 5 #글자수를 동일하게 만들어야 한다.
# # hidden_size = 5
# #
# x_data = []
# for seq in input_data:
#     x_data.append([word2idx[c] for c in seq])
# x_max_size = 0
# for x in x_data:
#     x_len = len(x)
#     if x_max_size < x_len:
#         x_max_size = x_len
#
# y_data = [[word2idx[c] for c in target_data]]
#
#
# input_dim = x_max_size
# sequence_length = 3
# hidden_size = 3
#
# X = tf.placeholder(tf.int32, [None, sequence_length])
# x_one_hot = tf.one_hot(X, input_dim)
# Y = tf.placeholder(tf.int32, [None, sequence_length])
#
# cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
# initial_state = cell.zero_state(batch_size, tf.float32)
# outputs, _state = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
#
#
# weights = tf.ones([batch_size, sequence_length])
# sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)
# prediction = tf.argmax(outputs, axis=2)
#
# print(x_one_hot)
# print(y_data)
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(20001):
#         l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
#         result = sess.run(prediction, feed_dict={X: x_one_hot})
#         print(i, "loss:", l, "prediction: ", result, "true Y:", y_data)
#
#         result_str = [idx2word[c] for c in np.squeeze(result)]
#         print("\tPrediction str:", ''.join(result_str))
#
