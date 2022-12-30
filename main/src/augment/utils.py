#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


def remove_duplicates(xs_list, ys_list):
    return map(list, zip(*list(set([(x, y) for x, y in zip(xs_list, ys_list)]))))

def get_lcs_table(x: list, y: list, reverse: bool=False) -> (int, int, int, list):
    """
    Build lcs_table in bottom up fashion.
    """
    # if reverse
    if reverse:
        x, y = x[::-1], y[::-1]
    # max length of lcs
    l = min(len(x), len(y))
    # initialize a table with zeros
    # for the length of longest common suffix
    lcs_table = [[0 for _ in range(l+1)] for _ in range(l+1)]
    # the longest length
    lcs_len = 0
    # the index of the maximum value
    max_i, max_j = 0, 0
    # build lcs_table in bottom up fashion
    for i, j in zip(range(l+1), range(l+1)):
        if i*j == 0:
            lcs_table[i][j] = 0
        elif x[i-1] == y[j-1]:
            lcs_table[i][j] = lcs_table[i-1][j-1] + 1
            lcs_len = max(lcs_len, lcs_table[i][j])
            if lcs_len == lcs_table[i][j]:
                max_i, max_j = i, j
        else:
            break
    return lcs_len, max_i, max_j, lcs_table

def get_lcs(x: list, y:list) -> list:
    """
    Dynamic Programming to find the longest common substring in O(m*n) time.
    """
    # left to right
    lcs_len, max_i, max_j, lcs_table = get_lcs_table(x, y)
    left_lcs = trace_lcs(x, lcs_len, max_i, max_j, lcs_table)
    # right to left
    lcs_len, max_i, max_j, lcs_table = get_lcs_table(x, y, True)
    right_lcs = trace_lcs(x, lcs_len, max_i, max_j, lcs_table, True)
    if len(left_lcs) < len(right_lcs):
        return right_lcs
    else:
        return left_lcs

def trace_lcs(x, lcs_len, max_i, max_j, lcs_table, reverse=False):
    """
    Trace back the longest common substring given DP table.
    """
    # if reverse
    if reverse:
        x = x[::-1]
    # trace back diagonally
    lcs = []
    if lcs_len:
        while lcs_table[max_i][max_j]:
            lcs.append(x[max_i-1])
            max_i -= 1
            max_j -= 1
    if reverse:
        return lcs
    else:
        return lcs[::-1]

def detokenize(x: list, tokenizer) -> str:
    # detokenize list into string
    idx_x = tokenizer.convert_tokens_to_ids(x)
    return tokenizer.decode(idx_x)

def linear_decompose(x: str, y: str, tokenizer, low=1) -> (str, str):
    # tokenization
    tk_x, tk_y = tokenizer.tokenize(x), tokenizer.tokenize(y)
    # get the lcs of x and y via DP
    lcs = get_lcs(tk_x, tk_y)
    lcs_len = len(lcs)
    if lcs_len > low:
        if ' '.join(tk_x).startswith(' '.join(lcs)):
            tk_x, tk_y = tk_x[lcs_len:], tk_y[lcs_len:]
        else:
            tk_x, tk_y = tk_x[:lcs_len], tk_y[:lcs_len]
        if len(tk_x) > low and len(tk_y) > low:
            return detokenize(tk_x, tokenizer), detokenize(tk_y, tokenizer)
    return '', ''