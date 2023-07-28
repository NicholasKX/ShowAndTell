# -*- coding: utf-8 -*-
"""
Created on 2023/7/28 15:25 
@Author: Wu Kaixuan
@File  : train_npu.py 
@Desc  : train_npu 
"""
import os.path
import time

import mindspore
import mindspore.nn as nn
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
from mindspore.nn.optim import optimizer
from tqdm import tqdm
from networks.resnet50_lstm import DecoderRNN, EncoderCNN, CNN2RNN
from dataset import FlickrDataset
from mindspore.dataset import GeneratorDataset
import argparse
import numpy as np
import mindspore as ms
import os
from mindspore import context
from openi import openi_multidataset_to_env as DatasetToEnv
from openi import env_to_openi,obs_copy_file
from openi import pretrain_to_env
os.system('pip install nltk -i https://pypi.douban.com/simple')



def seed_everything(seed):
    if seed:
        ms.set_seed(seed)
        np.random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser(description='Train the Show and Tell network')

    parser.add_argument('--data_url',
                        help='path to training/inference dataset folder',
                        default='./data')
    parser.add_argument('--multi_data_url', help='path to training/inference dataset folder',
                        default='./data')
    parser.add_argument('--train_url',
                        help='model folder to save/load',
                        default='./model')
    parser.add_argument('--pretrain_url',
                        help='model folder to save/load',
                        default='./pretrain')
    parser.add_argument('--result_url',
                        help='folder to save inference results',
                        default='./result')

    parser.add_argument('--dataset_path', type=str, default="/cache/data", help='Dataset path.')
    parser.add_argument('--model_saved_path', default="/cache/model_saved", type=str, help='')
    parser.add_argument('--pretrained_model_path', default="/cache/pretrained_model", type=str, help='pretrained model directory')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    # parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute.')
    parser.add_argument('--device_num', type=int, default=1, help='Device num.')
    parser.add_argument('--device_target', type=str, default="Ascend", help='Device choice Ascend or GPU')
    # parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
    # parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=20, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataset workers num.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
    args, _ = parser.parse_known_args()
    return args


def init_env(args):
    work_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"WORK DIR:{work_dir}")
    seed_everything(args.seed)
    device_id = int(os.getenv('DEVICE_ID', 0))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id,
                        max_device_memory="30GB", )


def create_dataset_flickr(data,
                          usage,
                          resize: int = 224,
                          batch_size: int = 1,
                          workers: int = 1):
    print(f"Creating {usage} dataset...")
    dataset = GeneratorDataset(data, ["image", "caption"])
    trans = []
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


def main(args):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    img_root = os.path.join(args.dataset_path, os.listdir(args.dataset_path)[0],"Flicker8k_Dataset")
    print(f"IMG ROOT:{img_root}")
    model_save_path = args.model_saved_path
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    init_env(args)
    dataset_dir = os.path.join(abs_path, "flickr8k")
    print(f"DATASET DIR:{dataset_dir}")
    data_train = FlickrDataset(dataset_dir, img_dir=img_root, split="train")
    data_test = FlickrDataset(dataset_dir, img_dir=img_root, split="test")
    vocab_size = len(data_train.vocab)
    print(f"Avg len:{data_train.get_avg_len()}")
    print(f"Vocabulary size: {len(data_train.vocab)}")
    # 获取处理后的训练与测试数据集
    train_data = create_dataset_flickr(data_train, usage="train", resize=224,
                                       batch_size=args.batch_size, workers=args.num_workers)
    step_size_train = train_data.get_dataset_size()
    print(f"step_size_train: {step_size_train}")
    test_data = create_dataset_flickr(data_test, usage="test", resize=224,
                                      batch_size=args.batch_size, workers=args.num_workers)
    step_size_test = test_data.get_dataset_size()
    print(f"step_size_test: {step_size_test}")
    # 创建迭代器
    data_loader_train = train_data.create_tuple_iterator(num_epochs=args.epochs)
    data_loader_val = test_data.create_tuple_iterator(num_epochs=args.epochs)

    # 设置学习率
    # lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size_train * num_epochs,
    #                         step_per_epoch=step_size_train, decay_epoch=num_epochs)
    # 创建模型
    pretrained_ckpt = os.path.join(args.pretrained_model_path, "resnet50_224_new.ckpt")
    print(f"PRETRAINED CKPT:{pretrained_ckpt}")
    model = CNN2RNN(embed_size=300, hidden_size=512, vocab_size=vocab_size, num_layers=1,
                    resnet50_ckpt=pretrained_ckpt)

    # 定义优化器和损失函数
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=args.lr)
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
        for images, captions in tqdm(data_loader_val, total=step_size_test):
            logits = model(images, captions[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), captions.reshape(-1))
            loss_total += loss.asnumpy()
            step_total += 1
        loss_test = loss_total / step_total
        print(f"Epoch {epoch} loss={loss_total / step_total}")
        return loss_test

    for i in tqdm(range(args.epochs)):
        train_loop(model, data_loader_train, i)
        loss_test = evaluate(model, data_loader_val, i)
        mindspore.save_checkpoint(model,
                                  os.path.join(model_save_path, f"model_{i}_loss_{loss_test}.ckpt"))


if __name__ == '__main__':
    args = get_parser()
    print(args)
    data_dir = '/cache/data'
    train_dir = '/cache/model_saved'
    model_dir = '/cache/pretrained_model'
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    except Exception as e:
        print("path already exists")
    pretrain_to_env(args.pretrain_url, model_dir)
    DatasetToEnv(args.multi_data_url, data_dir)
    try:
        print("List files in data_dir: {}".format(os.listdir(data_dir)))
        print("List files in dataset_dir: {}".format(len(os.listdir(os.path.join(data_dir, os.listdir(data_dir)[0])))))
    except Exception as e:
        print(str(e))
    start = time.time()
    main(args)
    end = time.time()
    print("training time: ", end - start)
    env_to_openi(train_dir, args.train_url)

