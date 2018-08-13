from konlpy.tag import Okt
import numpy as np

####################################################
# 데이터 konlpy를 이용한 토큰나이즈                #
####################################################


def load_data(data):
    okt = Okt()
    question = []
    answer = []
    for seq in data:
        if type(seq[0]) is not str or type(seq[1]) is not str:
            continue
        if len(seq[0]) > 0 and len(seq[0]) > 0:
            q_tmp = okt.morphs(seq[0])
            a_tmp = okt.morphs(seq[1])
            question.append(q_tmp)
            answer.append(a_tmp)
    return question, answer

####################################################
# 데이터 말뭉치 사전화                             #
####################################################


def make_dic(contents):
    words = []
    for seq in contents:
        for word in seq:
            words.append(word)
    words.append('<PAD>')
    words.append('<S>')
    words.append('<E>')
    words.append('<UNK>')

    idx_to_word = {idx: word for idx, word in enumerate(words)}
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    #print('컨텐츠 갯수 : %s, 단어 갯수 : %s'% (len(contents), len(idx_to_word)))
    return word_to_idx, idx_to_word


####################################################
# 문장의 길이 체크                                 #
####################################################

def check_seqlength(docs):
    max_document_length = 0
    for doc in docs:
        document_length = len(doc)
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length


####################################################
# inputs 데이터 생성                                #
####################################################


def make_inputs(encoder_in, decoder_in, word_to_idx, encoder_size, decoder_size):
    rawinputs = np.array(encoder_in)
    rawtargets = np.array(decoder_in)

    encoder_input = []
    decoder_input = []
    targets = []
    target_weights = []

    for rawinput, rawtarget in zip(rawinputs, rawtargets):
        # 인코더
        tmp_encoder_input = [word_to_idx[c] for idx, c in enumerate(rawinput) if idx < encoder_size and c in word_to_idx]
        encoder_pad_size = max(encoder_size - len(tmp_encoder_input), 0)
        encoder_pad = [word_to_idx['<PAD>']] * encoder_pad_size
        encoder_input.append(tmp_encoder_input + encoder_pad)

        # 디코더
        tmp_decoder_input = [word_to_idx[c] for idx, c in enumerate(rawtarget) if idx < encoder_size and c in word_to_idx]
        decoder_pad_size = max(decoder_size - len(tmp_encoder_input), 0)
        decoder_pad = [word_to_idx['<PAD>']] * decoder_pad_size
        decoder_input.append([word_to_idx['<S>']] + tmp_decoder_input + decoder_pad)

        # 타겟
        targets.append(tmp_decoder_input + [word_to_idx['<E>']] + decoder_pad)

        # 타겟 weight
        tmp_targets_weight = np.ones(decoder_size, dtype=np.float32)
        tmp_targets_weight[-decoder_pad_size:] = 0
        target_weights.append(tmp_targets_weight)

    return encoder_input, decoder_input, targets, target_weights


def make_batch(encoder_inputs, decoder_inputs, targets, target_weights):

    encoder_size = len(encoder_inputs[0])
    decoder_size = len(decoder_inputs[0])
    encoder_inputs, decoder_inputs, targets, target_weights = np.array(encoder_inputs), np.array(decoder_inputs), np.array(targets), np.array(target_weights)
    result_encoder_inputs = []
    result_decoder_inputs = []
    result_targets = []
    result_target_weights = []

    for i in range(encoder_size):
        result_encoder_inputs.append(encoder_inputs[:i])
    for j in range(decoder_size):
        result_decoder_inputs.append(decoder_inputs[:j])
        result_targets.append(targets[:j])
        result_target_weights.append(target_weights[:j])
    return result_encoder_inputs, result_decoder_inputs, result_targets, result_target_weights
