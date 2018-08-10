import tensorflow as tf
import math


sess = tf.InteractiveSession()
tf.set_random_seed(777)  # reproducibility

# linear regression 교육을 시킨다
chat_log = {"당뇨병의 정의": "당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다. 포도당은 우리가 먹는 음식물 중 탄수화물의 기본 구성성분입니다.",
            "당뇨병의 원인": "유전적 요인 환경적 요인",
            "당뇨병의 증상": "혈당이 높아지면 소변으로 당이 빠져나가게 되는데, 이때 포도당이 다량의 물을 끌고 나가기 때문에 소변을 많이 보게 됩니다. 따라서 몸 안의 수분이 모자라 갈증이 심하며 물을 많이 마시게 됩니다. 또한, 우리가 섭취한 음식물이 소변으로 빠져나가 에너지로 이용되지 못하므로 공복감은 심해지고 점점 더 먹으려 합니다.",
            "당뇨병의 분류": "제 1형 당뇨병 제 2형 당뇨병 기타 형태의 당뇨병 임신성 당뇨병",
            "당뇨병의 진단": "당뇨병의 증상은 다양하며 때로는 전혀 증상이 없는 경우도 있습니다.",
            "당뇨병이란?": "인슐린은 췌장 랑게르한스섬에서 분비되어 식사 후 올라간 혈당을 낮추는 기능을 합니다. "}

# 원본 데이터
question_context = chat_log.keys()
answer_context = chat_log.values()

question_list = {n: c for n, c in enumerate(question_context)}
answer_list = {n: c for n, c in enumerate(answer_context)}

# X and Y data
x_data = list(question_list.keys())
y_data = list(answer_list.keys())

# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let TensorFlow figure it out
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# X 와 Y 의 상관 관계를 분석하기 위한 가설 수식을 작성합니다.
# y = W * x + b
# W 와 X 가 행렬이 아니므로 tf.matmul 이 아니라 기본 곱셈 기호를 사용했습니다.
hypothesis = W * X + b

# 손실 함수를 작성합니다.
# mean(h - Y)^2 : 예측값과 실제값의 거리를 비용(손실) 함수로 정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 비용을 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(100):
        # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    result = sess.run(hypothesis, feed_dict={X: 2})
    print(answer_list[math.ceil(result)])
    result = sess.run(hypothesis, feed_dict={X: 3})
    print(answer_list[math.ceil(result)])