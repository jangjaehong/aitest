import data_func
from konlpy.tag import Okt
import numpy as np
import tensorflow as tf
import time


# 입력의 형태 shape = (batch_size, squence_length, dim_length)
# batch_data 만들시 주의사항
# batch_size: 개발의 임의로,
# sequece_length: encoder와 decoder는 갯수가 같아야 한다.
# dim_lengt: encoder 각각 sequence들의 길이는 같아야 한다. decoder도 마찬가지
# 하지만 sequence들은 대부분 길이가 같을수가 없다. 그래서 sequence들중 max_length를 찾고,
# 길이가 짧을 시 max_length와 현재 sequence 길의 차이만큼 <PAD>를 추가해준다.


twitter = Okt()
chat_log = [
    ["당뇨병의 정의",
     "당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다."
     "포도당은 우리가 먹는 음식물 중 탄수화물의 기본 구성성분입니다. "
     "탄수화물은 위장에서 소화효소에 의해 포도당으로 변한 다음 혈액으로 흡수됩니다. "
     "흡수된 포도당이 우리 몸의 세포들에서 이용되기 위해서는 인슐린이라는 호르몬이 반드시 필요합니다."
     "인슐린은 췌장 랑게르한스섬에서 분비되어 식사 후 올라간 혈당을 낮추는 기능을 합니다."
     "만약 여러 가지 이유로 인하여 인슐린이 모자라거나 성능이 떨어지게 되면, 체내에 흡수된 포도당은 이용되지 못하고 혈액 속에 쌓여 소변으로 넘쳐 나오게 되며, 이런 병적인 상태를 '당뇨병' 이라고 부르고 있습니다."
     "우리나라도 최근 들어 사회 경제적인 발전으로 과식, 운동부족, 스트레스 증가 등으로 인하여 당뇨병 인구가 늘고 있습니다. "
     "2010년 통계를 보면 우리나라의 전체 인구 중 350만명 정도가 당뇨병환자인 것으로 추정되고 있으나, 이중의 반 이상은 아직 자신이 당뇨병환자임을 모르고 지냅니다."],
    ["당뇨병의 원인", "유전적 요인, 환경적 요인"],
    ["유전적 요인",
     "당뇨병의 발병 원인은 아직 정확하게 규명이 되어있지 않습니다. "
     "현재까지 밝혀진 바에 의하면 유전적 요인이 가장 가능성이 큽니다."
     "만약, 부모가 모두 당뇨병인 경우 자녀가 당뇨병이 생길 가능성은 30% 정도이고, 한 사람만 당뇨병인 경우는 15% 정도입니다."
     "하지만 유전적 요인을 가지고 있다고 해서 전부 당뇨병환자가 되는 것은 아니며, 유전적인 요인을 가진 사람에게 여러 가지 환경적 요인이 함께 작용하여 당뇨병이 생기게 됩니다."],
    ["환경적 요인",
     "비 만 - 뚱뚱하면 일단 당뇨병을 의심하라는 말이 있듯이 비만은 당뇨병과 밀접한 관련이 있습니다. "
     "계속된 비만은 몸 안의 인슐린 요구량을 증가시키고, 그 결과로 췌장의 인슐린 분비기능을 점점 떨어뜨려 당뇨병이 생깁니다. "
     "또한 비만은 고혈압이나 심장병의 원인이 되기도 합니다. "
     "연령 - 당뇨병은 중년 이후에 많이 발생하며 연령이 높아질수록 발병률도 높아집니다. "
     "식생활 - 과식은 비만의 원인이 되고, 당뇨병을 유발하므로 탄수화물(설탕포함)과 지방의 과다한 섭취는 피해야 합니다. "
     "운동부족 - 운동부족은 고혈압, 동맥경화 등 성인병의 원인이 됩니다. "
     "운동부족은 비만을 초래하고, 근육을 약화시키며, 저항력을 저하시킵니다. "
     "스트레스 - 우리 몸에 오래 축적된 스트레스는 부신피질호르몬의 분비를 증가시키고, 저항력을 떨어뜨려 질병을 유발합니다."
     "성별 - 일반적으로 여성이 남성보다 발병률이 높습니다.그 이유는 임신이라는 호르몬 환경의 변화 때문입니다."
     "호르몬 분비 당뇨병과 직접 관련이 있는 인슐린과 글루카곤 호르몬에 이상이 생기면 즉각적으로 당뇨병이 유발되며, 뇌하수체나 갑상선, 부신호르몬과 같은 간접적인 관련인자도 당뇨병을 일으킬 수 있습니다."
     "감염증 감염증에 걸리면 신체의 저항력이 떨어지고, 당대사도 나빠지게 되어 당뇨병이 발생하기 쉽습니다.특히 췌장염, 간염,담낭염 등은 당뇨병을 일으킬 가능성이 크므로 신속하게 치료해야 합니다."
     "약물복용 다음과 같은 약물을 장기간 사용하는 경우에는 당뇨병 소질을 갖고 있는 사람에게 영향을 끼칠 수 있습니다."
     "호르몬 분비 - 당뇨병과 직접 관련이 있는 인슐린과 글루카곤 호르몬에 이상이 생기면 즉각적으로 당뇨병이 유발되며, 뇌하수체나 갑상선, 부신호르몬과 같은 간접적인 관련인자도 당뇨병을 일으킬수 있습니다."
     "감염증 - 감염증에 걸리면 신체의 저항력이 떨어지고, 당대사도 나빠지게 되어 당뇨병이 발생하기 쉽습니다.특히 췌장염, 간염, 담낭염 등은 당뇨병을 일으킬 가능성이 크므로 신속하게 치료해야 합니다."
     "약물복용 - 다음과 같은 약물을 장기간 사용하는 경우에는 당뇨병 소질을 갖고 있는 사람에게 영향을 끼칠 수 있습니다."
     "① 신경통, 류마티즘, 천식, 알레르기성 질환 등에 사용하는 부신피질 호르몬제"
     "② 혈압을 내리고 이뇨작용을 하는 강압 이뇨제"
     "③ 경구용 피임약"
     "④ 소염 진통제"
     "⑤ 갑상선 호르몬제 외과적 수술 위절제 수술 후 당대사에 이상이 생기는 경우가 있습니다."
     "따라서 위절제 수술을 받은 사람이면서, 당뇨병 소질을 갖고 있는 경우는 혈당의 변동을 주의  깊게 살펴야 합니다."
     "외과적 수술 - 위절제 수술 후 당대사에 이상이 생기는 경우가 있습니다."
     "따라서 위절제 수술을 받은 사람이면서, 당뇨병 소질을 갖고 있는 경우는 혈당의 변동을 주의 깊게 살펴야 합니다."],
    ["당뇨병의 증상",
     "혈당이 높아지면 소변으로 당이 빠져나가게 되는데, 이때 포도당이 다량의 물을 끌고 나가기 때문에 소변을 많이 보게 됩니다. "
     "따라서 몸 안의 수분이 모자라 갈증이 심하며 물을 많이 마시게 됩니다. 또한, 우리가 섭취한 음식물이 소변으로 빠져나가 에너지로 이용되지 못하므로 공복감은 심해지고 점점 더 먹으려 합니다."
     "당뇨병의 3대 증상은 다음(多飮), 다식(多食), 다뇨(多尿)이지만 이외에도 여러 증상이 있습니다. 당뇨병은 특별한 증상이 없을 수도 있어, 자신이 당뇨병인지 모르고 지내다가 뒤늦게 진단받는 경우도 있습니다."],
    ["당뇨병의 분류", "제 1형 당뇨병, 제 2형 당뇨병, 기타 형태의 당뇨병, 임신성 당뇨병"],
    ["제 1형 당뇨병",
     "당뇨병 우리나라 당뇨병의 2% 미만을 차지하며 주로 소아에서 발생하나, 성인에서도 나타날 수 있습니다."
     "급성 발병을 하며 심한 다음, 다뇨, 체중감소 등과 같은 증상들이 나타나고, 인슐린의 절대적인 결핍으로 인하여 케톤산증이 일어납니다."
     "고혈당의 조절 및 케톤산증에 의한 사망을 방지하기 위해 인슐린치료가 반드시 필요합니다."],
    ["제 2형 당뇨병",
     "당뇨병 한국인 당뇨병의 대부분을 차지하며 체중정도에 따라서 비만형과 비비만형으로 나눕니다."
     "생활수준의 향상으로 칼로리의 과잉섭취가 많거나 상대적으로 운동량이 감소하고 많은 스트레스에 노출되면 인슐린의 성능이 떨어져서 당뇨병이 발현되며 계속 조절하지 않을 경우 인슐린 분비의 감소가 따르게 됩니다."
     "주로 40세 이후에 많이 발생하고 반 수 이상의 환자가 과체중이거나 비만증을 갖고 있습니다."
     "제1형 당뇨병에 비해 임상증상이 뚜렷하지 않고 가족성 경향이 있으며, 특수한 경우 이외에는 케톤산증과 같은 급성 합병증을 일으키지 않고 초기에 식사와 운동요법에 의하여 체중을 감량하고 근육을 키우면 당뇨병이 호전되는 경우가 많습니다."],
    ["기타 형태의 당뇨병", "췌장질환, 내분비질환, 특정한 약물, 화학물질, 인슐린 혹은 인슐린 수용체 이상, 유전적 증후군에 의해 2차적으로 당뇨병이 유발되는 경우가 있습니다."],
    ["임신성 당뇨병",
     "임신성 당뇨병이란 임신 중 처음 발견되었거나 임신의 시작과 동시에 생긴 당조절 이상을 말하며 임신 전 진단된 당뇨병과는 구분됩니다."
     "임산부의 2∼3%가 발병하며, 대부분은 출산 후 정상화됩니다."
     "하지만 임신 중에는 혈당조절의 정도가 정상범위를 벗어나는 경우 태아 사망률 및 선천성 기형의 이환율이 높으므로 주의를 요합니다."
     "당뇨병의 가족력이 있거나 거대아, 기형아, 사산아를 출산한 분만력이 있는 경우, 그리고 산모가 비만한 경우, 고혈압이 있거나 요당이 나오는 경우는 보통 임신 24주∼28주에 간단한 임신성 당뇨병 검사를 받아야 합니다."],
    ["당뇨병의 진단",
     "당뇨병의 증상은 다양하며 때로는 전혀 증상이 없는 경우도 있습니다."
     "따라서 다음과 같은 경우에는 당뇨병에 대한 검사를 해 보는 것이 좋습니다."
     "① 연령·체형 40세 이상으로 비만한 사람"
     "② 가족력 가까운 친척 중에서 당뇨병이 있는 사람"
     "③ 자각증상 갈증, 다음, 다뇨, 다식, 피로감, 체중감소 등의 증상이 있는 사람"
     "④ 당뇨병이 합병되기 쉬운 질환이 있는 사람 고혈압, 췌장염, 내분비 질환, 담석증"
     "⑤ 당뇨병 발병을 촉진하는 약물을 사용하고 있는 사람 다이아자이드계 혈압 강하제나 신경통에 쓰이는 부신피질 호르몬 인 스테로이드 제품을 장기간 복용하는 사람"],
]

#################################
# 데이터를 학습 시키기위한 작업 #
#################################
question, answer = data_func.load_data(chat_log)
word_to_idx, idx_to_word = data_func.make_dic(contents=question + answer)
# parameters
vocab_size = len(idx_to_word) # 말뭉치 단어의 수
learning_rate = 0.001 # 학습률
batch_size = 16 # 1회당 학습하는 데이터 수(배치크기)
hidden_size = 300
encoder_size = data_func.check_seqlength(question) # 인코더에 넣을 입력 시퀸스의 최대 길이, 4
decoder_size = 100 # 디코더에 넣을 시퀸스의 최대 길이, 548
steps_per_checkpoint = 10
# 인코더 : 질문
# 디코더 : 답변
# 타켓 : 답변
encoderinputs, decoderinputs, targets_, targetweights = \
    data_func.make_inputs(question, answer, word_to_idx,encoder_size=encoder_size, decoder_size=decoder_size, )

source_vocab_size = vocab_size
target_vocab_size = vocab_size
batch_size = batch_size
encoder_size = encoder_size
decoder_size = decoder_size
learning_rate = tf.Variable(float(learning_rate), trainable=False)
global_step = tf.Variable(0, trainable=False)

W = tf.Variable(tf.random_normal([hidden_size, vocab_size]))
b = tf.Variable(tf.random_normal([vocab_size]))
output_projection = (W, b)

encoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
decoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(decoder_size)]
targets = [tf.placeholder(tf.int32, [batch_size]) for _ in range(decoder_size)]
target_weights = [tf.placeholder(tf.float32, [batch_size]) for _ in range(decoder_size)]

cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
    encoder_inputs, decoder_inputs, cell,
    num_encoder_symbols=vocab_size,
    num_decoder_symbols=vocab_size,
    embedding_size=hidden_size,
    output_projection=output_projection,
    feed_previous=True)
logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in outputs]


def step(session, encoderinputs, decoderinputs, targets, target_weights):
    input_feed = {}
    for l in range(len(encoder_inputs)):
        input_feed[encoder_inputs[l].names] = encoderinputs[l]
    for l in range(decoder_inputs):
        input_feed[decoder_inputs[l].names] = decoderinputs[l]
        input_feed[targets[l].names] = targets[l]
        input_feed[target_weights[l].names] = targetweights[l]
        output_feed = []
        for l in range(len(decoder_inputs)):
            output_feed.append(logits[l])
    output = session.run(output_feed, input_feed)
    return output[0:]  # outputs


sess = tf.Session()
sess.run(tf.global_variables_initializer())
step_time, loss = 0.0, 0.0
current_step = 0
start = 0
end = batch_size
while current_step < 10000:
    # if end > len(answer):
    #     start = 0
    #     end = batch_size

        # Get a batch and make a step
    print(decoder_inputs[start:end])

    start_time = time.time()
    encoder_inputs, decoder_inputs, targets, target_weights = data_func.make_batch(encoderinputs[start:end],
                                                                                   decoderinputs[start:end],
                                                                                   targets_[start:end],
                                                                                   targetweights[start:end])
    if current_step % steps_per_checkpoint == 0:
        for i in range(decoder_size - 2):
            decoder_inputs[i + 1] = np.array([word_to_idx['<PAD>']] * batch_size)
        output_logits = step(sess, encoder_inputs, decoder_inputs, targets, target_weights)
        predict = [np.argmax(logit, axis=1)[0] for logit in output_logits]
        predict = ' '.join(idx_to_word[ix][0] for ix in predict)
        real = [word[0] for word in targets]
        real = ' '.join(idx_to_word[ix][0] for ix in real)
        print('\n----\n step : %s \n time : %s \n LOSS : %s \n 예측 : %s \n 손질한 정답 : %s \n 정답 : %s \n----' %
              (current_step, step_time, loss, predict, real, answer[start]))
        loss, step_time = 0.0, 0.0