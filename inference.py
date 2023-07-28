# -*- coding: utf-8 -*-
"""
Created on 2023/7/28 12:33 
@Author: Wu Kaixuan
@File  : inference.py 
@Desc  : inference 
"""
import argparse
import os
import re
import mindspore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from PIL import Image
from tqdm import tqdm
from networks.resnet50_lstm import DecoderRNN, EncoderCNN, CNN2RNN
from dataset import FlickrDataset
from mindspore.dataset import GeneratorDataset
from mindspore import load_checkpoint, load_param_into_net
from dataset import Vocabulary
from mindspore import Tensor
from nltk.translate.bleu_score import sentence_bleu
from dataset import TestDatasetForBLEU


def inference(img_path, model, vocab):
    transform = transforms.Compose(
        [
            vision.Resize((224, 224)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            vision.HWC2CHW(),
        ]
    )

    test_img = Tensor(transform(Image.open(img_path).convert("RGB"))[0]).unsqueeze(0)

    print(
        "OUTPUT: "
        + " ".join(model.caption_image(test_img, vocab))
    )
    return model.caption_image(test_img, vocab)


def remove_punctuation_from_list(word_list):
    res = [re.sub(r'[^\w\s]', '', word) for word in word_list]
    return [word for word in res if word != '']


def get_bleu_score(dataset, model, vocab):
    data = dataset.get_data()  # json
    score = 0
    for key, value in data.items():
        img_path = os.path.join("flickr8k/Images", key)
        reference = value
        generated = inference(img_path, model, vocab)
        generated = remove_punctuation_from_list(generated)
        bleu = sentence_bleu(reference, generated, weights=(0.25, 0.25, 0.25, 0.25))
        score += bleu
    print(f"BLEU SCORE:{score / len(data)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Show and Tell network')
    parser.add_argument('--model_path',
                        help='path to model',
                        default='model_saved/model_4.ckpt')
    parser.add_argument('--img_path', help='path to image',
                        default='demo.jpg')

    args = parser.parse_args()
    vocab = Vocabulary(5)
    model = CNN2RNN(embed_size=300, hidden_size=512, vocab_size=len(vocab), num_layers=1,
                    resnet50_ckpt="pretrained_model/resnet50_224_new.ckpt")
    param_dict = mindspore.load_checkpoint(args.model_path)
    param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    # print(param_not_load)
    inference(img_path=args.img_path, model=model, vocab=vocab)
    # get_bleu_score(dataset, model, vocab)
