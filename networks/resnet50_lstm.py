# -*- coding: utf-8 -*-
"""
Created on 2023/7/27 23:48 
@Author: Wu Kaixuan
@File  : resnet50_lstm.py 
@Desc  : resnet50_lstm 
"""
from mindspore import Tensor
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore import load_checkpoint, load_param_into_net
from networks.resnet50 import resnet50
from mindspore import ops
from mindspore.common.initializer import HeUniform, Uniform
import math

class EncoderCNN(nn.Cell):
    def __init__(self,embed_size, train_cnn=False, resnet50_ckpt="pretrained_model/resnet50.ckpt"):
        super(EncoderCNN, self).__init__()
        self.trainCNN = train_cnn
        self.resnet50 = resnet50(pretrained=True,resnet50_ckpt=resnet50_ckpt)
        self.resnet50.fc = nn.Dense(self.resnet50.fc.in_channels, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def construct(self, images):
        features = self.resnet50(images)
        # for param in self.resnet50.trainable_params():
        #     if "fc.weight" in param.name or "fc.bias" in param.name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = self.trainCNN
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Cell):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_size))
        self.liner = nn.Dense(hidden_size, vocab_size,weight_init=weight_init,bias_init=bias_init)
        self.dropout = nn.Dropout(0.5)

    def construct(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = ops.concat((features.unsqueeze(1), embeddings), axis=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.liner(hiddens)
        return outputs


class CNN2RNN(nn.Cell):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_cnn=False, resnet50_ckpt=None):
        super(CNN2RNN, self).__init__()
        self.cnn = EncoderCNN(embed_size, train_cnn=train_cnn, resnet50_ckpt=resnet50_ckpt)
        self.rnn = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def construct(self, images, captions):
        features = self.cnn(images)
        outputs = self.rnn(features, captions)
        return outputs

    def caption_image(self, image, vocab, max_length=25):
        result = []
        x = self.cnn(image).unsqueeze(0)
        states = None
        for _ in range(max_length):
            hiddens, states = self.rnn.lstm(x, states)
            output = self.rnn.liner(hiddens.squeeze(0))
            predicted = output.argmax(1)
            result.append(int(predicted.asnumpy()))
            x = self.rnn.embed(predicted).unsqueeze(0)
            if vocab.itos[int(predicted.asnumpy())] == "<EOS>":
                break
        return [vocab.itos[idx] for idx in result]

if __name__ == '__main__':
    # 定义ResNet50网络
    pretrained_ckpt = "../pretrained_model/resnet50_224_new.ckpt"
    model = resnet50(num_classes=1000, pretrained=False)
    param_dict = load_checkpoint(pretrained_ckpt)
    load_param_into_net(model, param_dict)
    # 全连接层输入层的大小
    in_channel = model.fc.in_channels
    fc = nn.Dense(in_channels=in_channel, out_channels=10)
    # 重置全连接层
    model.fc = fc
    input = Tensor(np.ones([2, 3, 224, 224]), dtype=ms.float32)
    # 打印网络结构
    print(model)
    # 打印网络输出
    print(model(input))

    # 定义CNN2RNN网络
    model = CNN2RNN(embed_size=300,
                    hidden_size=512,
                    vocab_size=1004,
                    num_layers=1,
                    train_cnn=False,
                    resnet50_ckpt=pretrained_ckpt)

    image = Tensor(np.ones([2, 3, 224, 224]), dtype=ms.float32)
    caption = Tensor(np.ones([2, 10]), dtype=ms.int32)
    output = model(image, caption[:,:-1])
    print(output.shape)
