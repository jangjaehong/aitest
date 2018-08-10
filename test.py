import tensorflow as tf
from tensorflow.contrib import rnn
import os
import csv
import numpy as np
from konlpy.tag import Twitter

# linear regression 교육을 시킨다
chat_log = {"당뇨병 정의": ["당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다. 포도당은 우리가 먹는 음식물 중 탄수화물의 기본 구성성분입니다."],
            "당뇨병 원인": ["유전적 요인",  "환경적 요인"],
            "당뇨병 증상": ["혈당이 높아지면 소변으로 당이 빠져나가게 되는데, 이때 포도당이 다량의 물을 끌고 나가기 때문에 소변을 많이 보게 됩니다. 따라서 몸 안의 수분이 모자라 갈증이 심하며 물을 많이 마시게 됩니다. 또한, 우리가 섭취한 음식물이 소변으로 빠져나가 에너지로 이용되지 못하므로 공복감은 심해지고 점점 더 먹으려 합니다."],
            "당뇨병 분류": ["제 1형 당뇨병", "제 2형 당뇨병", "기타 형태의 당뇨병", "임신성 당뇨병"],
            "당뇨병 진단": ["당뇨병의 증상은 다양하며 때로는 전혀 증상이 없는 경우도 있습니다.", "당뇨병 판정 기준"]}


class Dictionary:
    file_list = []
    char_dic = []
    char_dic = {}
    char_len = 0

    # 가져올 파일이 있는 경로와 확장자 / mecab 사전 읽기 및 저장
    def __init__(self, path, extension):
        self.file_list = self.file_search(path, extension)
        self.char_set = self.open_csv(self.file_list)
        self.char_dic = {c: i for i, c in enumerate(self.char_set)}
        self.char_len = len(self.char_dic)

    # 디렉토리 파일 가져오기
    def file_search(self, path, extension):
        file_csv = []
        for (path, dir, files) in os.walk(path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == extension:
                    file_csv.append("%s/%s" % (path, filename))
        return file_csv

    # csv 파일 읽기
    def open_csv(self, files):
        chat_list = []
        # 엑셀파일 열기
        for file in files:
            # 앞서 디렉토리에서 가져온 csv 파일 열기
            open_file = open(file, 'r', encoding='utf-8')
            reader = csv.reader(open_file)
            for line in reader:
                chat_list.append(line[0])
            open_file.close()

        chat_list = set(chat_list)
        return chat_list

    # konlpy를 통한 질의 분석
    def parse_sentense(self, sentences):
        twitter = Twitter()
        # 단어구분을 통한 단순 분석
        morphs = []
        for sentence in sentences:
            parse = twitter.morphs(sentence)
            for char in parse:
                morphs.append(char)

        return list(set(morphs))


dictionary = Dictionary("D:/aitest/mecab-ko-dic", ".csv")
dictionary_dic = dictionary.char_dic

# 질의 분석 및 리스트로 저장
word_sequence = chat_log.keys() # 원본 데이터
word_set = parse_twitter(word_sequence) # 중복 제거 및 리스트로 반환 결과
word_dic = [dictionary_dic[c] for c in word_set] # 사전을 통해 만든 dic을 통해 one-hot 인코딩을 위한 넘버링

# hyper parameter
sequence_length = len(word_dic)
num_classes = len(word_dic)
hidden_size = len(word_dic)
batch_size = 5
learning_rate = 0.1

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

