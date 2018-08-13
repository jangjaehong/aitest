from tensorflow.contrib import rnn
import numpy as np
import tensorflow as tf
import te
from konlpy.tag import Okt
twitter = Okt()
chat_log = [
    ["당뇨병의 정의", "당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다. 포도당은 우리가 먹는 음식물 중 탄수화물의 기본 구성성분입니다."],
    ["당뇨병의 원인", "유전적 요인 환경적 요인"],
    ["당뇨병의 증상", "혈당이 높아지면 소변으로 당이 빠져나가게 되는데, 이때 포도당이 다량의 물을 끌고 나가기 때문에 소변을 많이 보게 됩니다. 따라서 몸 안의 수분이 모자라 갈증이 심하며 물을 많이 마시게 됩니다. 또한, 우리가 섭취한 음식물이 소변으로 빠져나가 에너지로 이용되지 못하므로 공복감은 심해지고 점점 더 먹으려 합니다."],
    ["당뇨병의 분류", "제 1형 당뇨병 제 2형 당뇨병 기타 형태의 당뇨병 임신성 당뇨병"],
    ["당뇨병의 진단", "당뇨병의 증상은 다양하며 때로는 전혀 증상이 없는 경우도 있습니다."],
    ["당뇨병이란?", "인슐린은 췌장 랑게르한스섬에서 분비되어 식사 후 올라간 혈당을 낮추는 기능을 합니다. "],
]

#################################
# 데이터를 학습 시키기위한 작업 #
#################################
q, a = te.load_data(chat_log)
word2idx, idx2word = te.make_dic(contents=q+a, minlength=.0, maxlength=3, jamo_delete=True)

# parameters
multi = True
forward_only = False
hidden_size = 300
vocab_size = len(idx2word)
num_layers = 3
learning_rate = 0.001
batch_size = 16
encoder_size = 100
decoder_size = tool.check_doclength(title,sep=True) # (Maximum) number of time steps in this batch
steps_per_checkpoint = 10
