#-*- coding:utf-8 _*-
"""
@author:fxw
@file: transform_label.py
@time: 2020/07/24
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
# import chardet
import numpy as np
import sys
import abc

def get_keys(key_path):
    with open(key_path,'r',encoding='utf-8') as fid:
        lines = fid.readlines()[0]
        lines = lines.strip('\n')
        return lines
    
#☯
class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, config):
        alphabet = get_keys(config['trainload']['key_file'])
        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text,t_step):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        result = []

        for i in range(len(text)):
            length.append(len(text[i]))
            for j in range(len(text[i])):
                index = self.dict[text[i][j]]
                result.append(index)

        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
    
    
    
class BaseConverter(object):

    def __init__(self, character):
        self.character = list(character)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    @abc.abstractmethod
    def train_encode(self, *args, **kwargs):
        '''encode text in train phase'''

    @abc.abstractmethod
    def test_encode(self, *args, **kwargs):
        '''encode text in test phase'''

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        '''decode label to text in train and test phase'''



class FCConverter(BaseConverter):

    def __init__(self, config):
        batch_max_length = config['base']['max_length']
        character = get_keys(config['trainload']['key_file'])
        self.character = character
        list_token = ['[s]']
        ignore_token = ['[ignore]']
        list_character = list(character)
        self.batch_max_length = batch_max_length + 1
        super(FCConverter, self).__init__(character=list_token + list_character + ignore_token)
        self.ignore_index = self.dict[ignore_token[0]]

    def encode(self, text):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.ignore_index)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            if self.batch_max_length>=len(text):
                batch_text[i][:len(text)] = torch.LongTensor(text)
            else:
                batch_text[i][:self.batch_max_length] = torch.LongTensor(text)[:self.batch_max_length]
        batch_text_input = batch_text
        batch_text_target = batch_text

        return batch_text_input, torch.IntTensor(length), batch_text_target

    def train_encode(self, text):
        return self.encode(text)

    def test_encode(self, text):
        return self.encode(text)

    def decode(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            texts.append(text)

        return texts