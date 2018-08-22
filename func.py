from collections import defaultdict
import operator
import numpy as np
from konlpy.tag import Okt

def load_data(sequence):
    okt = Okt()
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
    for seq in sequence:
        for word in seq:
            words.append(word)
    words.append('<PAD>')
    words.append('<S>')
    words.append('<E>')
    words.append('<UNK>')
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


def sequence_length(sequence):
    seq_length = []
    for seq in sequence:
        seq_length.append(len(seq))
    print("seq_length:", seq_length)
    return seq_length


def make_batch(rawencoder, rawdecoer, encoder_size, decoder_size, word2idx, dic_len):
    encoder_inputs = []
    decoder_inputs = []
    target_input = []
    for en_input, de_input in zip(rawencoder, rawdecoer):
        tmp_encoder_input = [word2idx[c] for c in en_input]
        encoder_pad_size = encoder_size - len(tmp_encoder_input)
        encoder_pad = [word2idx['<PAD>']] * encoder_pad_size
        encoder_inputs.append(np.eye(dic_len)[tmp_encoder_input + encoder_pad])

        tmp_decoder_input = [word2idx[c] for c in de_input]
        decoder_pad_size = decoder_size - len(tmp_decoder_input)
        decoder_pad = [word2idx['<PAD>']] * decoder_pad_size
        decoder_inputs.append(np.eye(dic_len)[[word2idx['<S>']] + tmp_decoder_input + decoder_pad])

        tmp_target_input = [word2idx[c] for c in de_input]
        target_pad_size = decoder_size - len(tmp_target_input)
        target_pad = [word2idx['<PAD>']] * target_pad_size
        target_input.append(tmp_decoder_input + target_pad + [word2idx['<E>']])

    return encoder_inputs, decoder_inputs, target_input

