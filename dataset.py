# -*- coding: utf-8 -*-
"""
Created on 2023/7/28 0:44 
@Author: Wu Kaixuan
@File  : dataset.py 
@Desc  : dataset 
"""
import re
import matplotlib.pyplot as plt
# import spacy  # for tokenizer
import os
from typing import *
import pandas as pd
from PIL import Image
import numpy as np
from PIL import Image
from mindspore.dataset import GeneratorDataset, MnistDataset
import json
from mindspore.dataset import text

# spacy_eng = spacy.load("en_core_web_sm")
tokenizer = text.BasicTokenizer(lower_case=True,
                                   keep_whitespace=False,
                                   preserve_unused_token=True,
                                   with_offsets=False)

def convert_json_key(param_dict):
    """
    json.dump不支持key是int的dict，在编码存储的时候会把所有的int型key写成str类型的
    """
    new_dict = dict()
    for key, value in param_dict.items():
        if isinstance(value, (dict,)):
            res_dict = convert_json_key(value)
            try:
                new_key = int(key)
                new_dict[new_key] = res_dict
            except:
                new_dict[key] = res_dict
        else:
            try:
                new_key = int(key)
                new_dict[new_key] = value
            except:
                new_dict[key] = value
    return new_dict


class Vocabulary:
    def __init__(self, freq_threshold, save_root="vocab"):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.save_root = save_root
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        else:
            try:
                with open(os.path.join(save_root, "word2idx.json"), 'r') as f:
                    self.stoi = json.load(f)
                    self.stoi = convert_json_key(self.stoi)
                with open(os.path.join(save_root, "idx2word.json"), 'r') as f:
                    self.itos = json.load(f)
                    self.itos = convert_json_key(self.itos)
            except Exception as e:
                print(e)
                print("can't load vocab")

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return tokenizer(text)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        self.save_vocab(os.path.join(self.save_root, "word2idx.json"),
                        os.path.join(self.save_root, "idx2word.json"))

    def tokenization(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

    def save_vocab(self, word2idx_path, idx2word_path):
        with open(word2idx_path, 'w') as f:
            json.dump(self.stoi, f)
        with open(idx2word_path, 'w') as f:
            json.dump(self.itos, f)


class FlickrDataset:
    def __init__(self, data_dir,img_dir,split="train", freq_threshold=5, transform=None):
        self.img_root = img_dir
        self.transform = transform
        txt_path = os.path.join(data_dir, "captions.txt")
        data = pd.read_csv(txt_path, sep=',')
        captions = data.caption
        self.avg_len = np.mean([len(caption.split(" ")) for caption in captions])
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(captions.tolist())
        if split == "train":
            self.data = pd.read_csv(os.path.join(data_dir, "train_captions.csv"))
            print(f"load train data from {os.path.join(data_dir, 'train_captions.csv')}")
        elif split == "test":
            self.data = pd.read_csv(os.path.join(data_dir, "test_captions.csv"))
            print(f"load test data from {os.path.join(data_dir, 'test_captions.csv')}")
        self.imgs = self.data.image
        self.captions = self.data.caption

    def __getitem__(self, index):
        img_path = os.path.join(self.img_root, self.imgs[index])
        caption = self.captions[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        tokenized_caption = [self.vocab.stoi["<SOS>"]]
        tokenized_caption += self.vocab.tokenization(caption)
        tokenized_caption.append(self.vocab.stoi["<EOS>"])
        return img, np.array(tokenized_caption)

    def __len__(self):
        return self.data.shape[0]

    def get_avg_len(self):
        return self.avg_len


def clean_text(text):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    return text


class TestDatasetForBLEU:
    def __init__(self, data_dir, freq_threshold=5):
        self.vocab = Vocabulary(freq_threshold)
        self.tokenizer = self.vocab.tokenizer_eng
        self.img_root = os.path.join(data_dir, "Images")
        self.data = pd.read_csv(os.path.join(data_dir, "test_captions.csv"))
        self.imgs = self.data.image
        self.captions = self.data.caption
        self.captions = self.captions.apply(clean_text)
        self.captions = self.captions.apply(self.tokenizer)
        # 把img相同的caption放在一起
        self.img2caption = {}
        for img, caption in zip(self.imgs, self.captions):
            if img not in self.img2caption:
                self.img2caption[img] = [caption]
            else:
                self.img2caption[img].append(caption)

    def get_data(self):
        return self.img2caption


if __name__ == '__main__':
    tokenizer = text.BasicTokenizer(lower_case=True,
                                   keep_whitespace=False,
                                   preserve_unused_token=True,
                                   with_offsets=False)
    text = "A child in a pink dress is climbing up a set of stairs in an entry way ."
    tokens = tokenizer(text)
    print(tokens)