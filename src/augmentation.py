from config import TOKEN_IDX
import numpy as np


def augment_none(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    x_aug.append(TOKEN_IDX[token_style]['UNK'])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    x_aug.append(TOKEN_IDX[token_style]['UNK'])
    y_aug.append(0)
    y_mask_aug.append(1)
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    return


def augment_sub_ins(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    r = np.random.randint(2)
    if r == 0:
        augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    else:
        augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)


def augment_sub_del(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    r = np.random.randint(2)
    if r == 0:
        augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    else:
        augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)


def augment_del_ins(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    r = np.random.randint(2)
    if r == 0:
        augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    else:
        augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)


def augment_all(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    r = np.random.randint(3)
    if r == 0:
        augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    elif r == 1:
        augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    else:
        augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)


AUGMENTATIONS = {
    'none': augment_none,
    'substitute': augment_substitute,
    'insert': augment_insert,
    'delete': augment_delete,
    'sub_ins': augment_sub_ins,
    'sub_del': augment_sub_del,
    'del_ins': augment_del_ins,
    'all': augment_all
}
