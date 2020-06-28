from config import TOKEN_IDX
import numpy as np


alpha_sub = 0.45
alpha_del = 0.45

tokenizer = None
# substitution strategy: 'unk' -> replace with unknown tokens, 'rand' -> replace with random tokens from vocabulary
sub_style = 'unk'


def augment_none(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    if sub_style == 'rand':
        x_aug.append(np.random.randint(tokenizer.vocab_size))
    else:
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


def augment_all(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    r = np.random.rand()
    if r < alpha_sub:
        augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    elif r < alpha_sub + alpha_del:
        augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    else:
        augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)


AUGMENTATIONS = {
    'none': augment_none,
    'substitute': augment_substitute,
    'insert': augment_insert,
    'delete': augment_delete,
    'all': augment_all
}
