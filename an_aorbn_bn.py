import rstr
import re
from random import randrange, randint

regexs = [
    '[ab]+',
]


def edge_case_neg(piece_len, off_by):
    """ generate edge cases """

    r = randint(0, 5)

    if r ==0:
        alen = piece_len + off_by
        ablen = piece_len - off_by
        blen = piece_len
    elif r == 1 :
        alen =piece_len - off_by
        ablen = piece_len + off_by
        blen = piece_len
    elif r == 2:
        alen = piece_len + off_by
        ablen = piece_len
        blen = piece_len - off_by
    elif r ==3:
        alen = piece_len - off_by
        ablen = piece_len
        blen = piece_len + off_by
    elif r == 4:
        alen = piece_len
        ablen = piece_len - off_by
        blen = piece_len + off_by
    else:
        alen = piece_len
        ablen = piece_len + off_by
        blen = piece_len - off_by

    an = 'a' * alen

    abn = ''
    while len(abn) < ablen:
        abn += rstr.xeger('[ab]*')
    abn = abn[:ablen]

    bn = 'b' * blen

    return an + abn + bn


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

    abn = ''
    while len(abn) < piece_len:
        abn += rstr.xeger('[ab]*')
    abn = abn[:piece_len]

    bn = 'b'*piece_len

    return an+abn+bn


with open('an-abn-bn-data.txt', 'w') as f:

    for regex in regexs:
        for i in range(11000):

            strlen = randrange(3, 30, 3)

            piece_len = strlen // 3
            if i % 5 == 0:
                f.write(get_pos(piece_len) + '\t1\n')
            else:
                if i % 3 == 0:
                    r = randint(1, 4)
                    neg = edge_case_neg(piece_len, r)
                else:
                    neg = get_neg(strlen)

                    if neg[:piece_len] == 'a' * piece_len and neg[piece_len * 2:] == 'b' * piece_len:
                        continue
                f.write(neg + '\t0\n')
