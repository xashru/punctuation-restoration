from config import TOKEN_IDX
import numpy as np


def augment_none(x, y, x_aug, y_aug, i, token_style):
    x_aug.append(x[i])
    y_aug.append(y[i])


def augment_substitute(x, y, x_aug, y_aug, i, token_style):
    x_aug.append(TOKEN_IDX[token_style]['UNK'])
    y_aug.append(y[i])


def augment_insert(x, y, x_aug, y_aug, i, token_style):
    x_aug.append(TOKEN_IDX[token_style]['UNK'])
    y_aug.append(0)
    x_aug.append(x[i])
    y_aug.append(y[i])


def augment_delete(x, y, x_aug, y_aug, i, token_style):
    return


def augment_sub_ins(x, y, x_aug, y_aug, i, token_style):
    r = np.random.randint(2)
    if r == 0:
        augment_substitute(x, y, x_aug, y_aug, i, token_style)
    else:
        augment_insert(x, y, x_aug, y_aug, i, token_style)


def augment_sub_del(x, y, x_aug, y_aug, i, token_style):
    r = np.random.randint(2)
    if r == 0:
        augment_substitute(x, y, x_aug, y_aug, i, token_style)
    else:
        augment_delete(x, y, x_aug, y_aug, i, token_style)


def augment_all(x, y, x_aug, y_aug, i, token_style):
    r = np.random.randint(3)
    if r == 0:
        augment_substitute(x, y, x_aug, y_aug, i, token_style)
    elif r == 1:
        augment_insert(x, y, x_aug, y_aug, i, token_style)
    else:
        augment_delete(x, y, x_aug, y_aug, i, token_style)


AUGMENTATIONS = {
    'none': augment_none,
    'substitute': augment_substitute,
    'insert': augment_insert,
    'delete': augment_delete,
    'sub_ins': augment_sub_ins,
    'sub_del': augment_sub_del,
    'all': augment_all
}
