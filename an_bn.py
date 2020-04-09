import rstr
import re
from random import randrange, randint

regexs = [
    '[ab]+',
]


def edge_case_neg(piece_len, off_by):
    """ generate edge cases """

    r = randint(0, 1)

    if r ==0:
        an = 'a' * (piece_len + off_by)
        bn = 'b' * (piece_len - off_by)
    else:
        an = 'a' * (piece_len - off_by)
        bn = 'b' * (piece_len + off_by)

    return an + bn


def get_neg(strlen):
    """ generate negative examples """
    neg = ''
    while len(neg) < strlen:
        neg += rstr.xeger(regex)
    neg = neg[:strlen]
    return neg


def get_pos(piece_len):
    """ generate positive examples """
    an = 'a'*piece_len

    bn = 'b'*piece_len

    return an+bn


with open('an-bn-data.txt', 'w') as f:

    for regex in regexs:
        for i in range(11000):

            strlen = randrange(3, 10, 2)

            piece_len = strlen // 2

            if i % 5 == 0:
                f.write(get_pos(piece_len) + '\t1\n')
            else:
                if i % 3 == 0:
                    r = randint(1, 4)
                    neg = edge_case_neg(piece_len, r)
                else:
                    neg = get_neg(strlen)

                    if neg[:piece_len] == 'a'*piece_len and neg[piece_len:] == 'b'*piece_len:
                        continue

                f.write(neg+'\t0\n')

