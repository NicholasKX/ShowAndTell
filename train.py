# -*- coding: utf-8 -*-
"""
Created on 2023/7/28 0:36 
@Author: Wu Kaixuan
@File  : train.py 
@Desc  : train 
"""
import os.path
from typing import Literal

import mindspore
import mindspore.nn as nn
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from PIL import Image
from mindspore import dtype as mstype
from mindspore.nn.optim import optimizer
from tqdm import tqdm

import dataset
from networks.resnet50_lstm import DecoderRNN, EncoderCNN, CNN2RNN
from dataset import FlickrDataset
from mindspore.dataset import GeneratorDataset
from mindspore import load_checkpoint, load_param_into_net
from nltk.translate.bleu_score import sentence_bleu



data_dir = "flickr8k/images"
batch_size = 4  # 批量大小
image_size = 32  # 训练图像空间大小
workers = 4  # 并行线程个数
num_classes = 10  # 分类数量
num_epochs = 5
model_save_path = "model_saved"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

def create_dataset_flickr(data,
                          usage: Literal["train", "test"] = "train",
                          resize: int = 224,
                          batch_size: int = 1,
                          workers: int = 1):
    dataset = GeneratorDataset(data, ["image", "caption"])
    trans = []
    if usage == "train":
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5)
        ]
    trans += [
        vision.Resize((resize, resize)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        vision.HWC2CHW()
    ]
    pad_op = transforms.PadEnd([25], pad_value=data.vocab.stoi["<PAD>"])
    target_trans = transforms.TypeCast(mstype.int32)
    # 数据映射操作
    dataset = dataset.map(operations=trans,
                          input_columns='image',
                          num_parallel_workers=workers)

    dataset = dataset.map(operations=[pad_op, target_trans],
                          input_columns='caption',
                          num_parallel_workers=workers)
    # 批量操作
    dataset = dataset.batch(batch_size)
    return dataset

if __name__ == '__main__':
    dataset_dir = "flickr8k"
    data_train = FlickrDataset(dataset_dir, split="train")
    data_test = FlickrDataset(dataset_dir, split="test")
    vocab_size = len(data_train.vocab)
    print(f"Avg len:{data_train.get_avg_len()}")
    print(f"Vocabulary size: {len(data_train.vocab)}")
    # 获取处理后的训练与测试数据集
    train_data = create_dataset_flickr(data_train, usage="train", resize=224, batch_size=batch_size, workers=workers)
    step_size_train = train_data.get_dataset_size()
    print(f"step_size_train: {step_size_train}")
    test_data = create_dataset_flickr(data_test, usage="test", resize=224, batch_size=batch_size, workers=workers)
    step_size_test = test_data.get_dataset_size()
    print(f"step_size_test: {step_size_test}")
    # 创建迭代器
    data_loader_train = train_data.create_tuple_iterator(num_epochs=num_epochs)
    data_loader_val = test_data.create_tuple_iterator(num_epochs=num_epochs)

    # 设置学习率
    lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size_train * num_epochs,
                            step_per_epoch=step_size_train, decay_epoch=num_epochs)
    # 创建模型
    model = CNN2RNN(embed_size=300, hidden_size=512, vocab_size=vocab_size, num_layers=1,
                    resnet50_ckpt="pretrained_model/resnet50_224_new.ckpt")

    # 定义优化器和损失函数
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')


    def forward_fn(images, captions):
        logits = model(images, captions[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), captions.reshape(-1))
        return loss


    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(images, captions):
        loss, grads = grad_fn(images, captions)
        optimizer(grads)
        return loss

    def train_loop(model, data_loader_train, epoch=0):
        model.set_train()
        loss_total = 0
        step_total = 0

        for batch, (images, captions) in tqdm(enumerate(data_loader_train), total=step_size_train):
            loss = train_step(images, captions)
            loss_total += loss.asnumpy()
            step_total += 1
            if batch % 100 == 0:
                print(f"Epoch {epoch} Batch {batch}, loss={loss_total / step_total}")


    def evaluate(model, data_loader_val, epoch=0):
        print("==========Evaluating==========")
        loss_total = 0
        step_total = 0
        model.set_train(False)
        for images, captions in tqdm(data_loader_val,total=step_size_test):
            logits = model(images, captions[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), captions.reshape(-1))
            loss_total += loss.asnumpy()
            step_total += 1
        print(f"Epoch {epoch} loss={loss_total / step_total}")


    for i in tqdm(range(num_epochs)):
        # train_loop(model, data_loader_train, i)
        evaluate(model, data_loader_val, i)
        mindspore.save_checkpoint(model, f"{model_save_path}/model_{i}.ckpt")
        # save_checkpoint(model, optimizer, i)
