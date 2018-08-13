from konlpy.tag import Okt
from collections import defaultdict
import operator

def load_data(data):
    question = []
    answer = []
    for seq in data:
        if type(seq[0]) is not str or type(seq[1]) is not str:
            continue
        if len(seq[0]) > 0 and len(seq[0]) > 0:
            q_tmp = ''.join(seq[0]).split()
            a_tmp = ''.join(seq[1]).split()
            question.append(q_tmp)
            answer.append(a_tmp)
    return question, answer


def make_dic(contents, minlength, maxlength, jamo_delete=False):
    dict = defaultdict(lambda: [])
    for doc in contents:
        for idx, word in enumerate(doc):
            if len(word) > minlength:
                normalizedword = word[:maxlength]
                if jamo_delete:
                    tmp = []
                    for char in normalizedword:
                        if ord(char) < 12593 or ord(char) > 12643:
                            tmp.append(char)
                    normalizedword = ''.join(char for char in tmp)
                if word not in dict[normalizedword]:
                    dict[normalizedword].append(word)
    dict = sorted(dict.items(), key=operator.itemgetter(0))[1:]
    words = []
    for i in range(len(dict)):
        word = []
        word.append(dict[i][0])
        for w in dict[i][1]:
            if w not in word:
                word.append(w)
        words.append(word)

    words.append(['<PAD>'])
    words.append(['<S>'])
    words.append(['<E>'])
    words.append(['<UNK>'])
    # word_to_ix, ix_to_word 생성
    ix_to_word = {i: ch[0] for i, ch in enumerate(words)}
    word_to_ix = {}
    for idx, words in enumerate(words):
        for word in words:
            word_to_ix[word] = idx
    print('컨텐츠 갯수 : %s, 단어 갯수 : %s'
          % (len(contents), len(ix_to_word)))
    return word_to_ix, ix_to_word

