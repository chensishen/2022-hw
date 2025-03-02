import argparse
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math

# 动态生成字符集
def build_vocab(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    return ''.join(sorted(set(text)))

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except ValueError:
            continue
    return tensor

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len - 1)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--chunk_len', type=int, default=200)
    argparser.add_argument('--batch_size', type=int, default=100)
    argparser.add_argument('--n_epochs', type=int, default=2000)
    args = argparser.parse_args()

    # 加载数据
    all_characters = build_vocab(args.filename)
    n_characters = len(all_characters)

    with open(args.filename, 'r', encoding='utf-8') as f:
        file = f.read()
    file_len = len(file)

    # 初始化模型
    hidden_size = 512
    decoder = CharRNN(n_characters, hidden_size, n_characters, model="gru", n_layers=2)
    if args.cuda:
        decoder.cuda()

    # 训练过程
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):
        inp, target = random_training_set(args.chunk_len, args.batch_size)
        hidden = decoder.init_hidden(args.batch_size)
        decoder.zero_grad()
        loss = 0
        for c in range(args.chunk_len):
            output, hidden = decoder(inp[:, c], hidden)
            loss += nn.CrossEntropyLoss()(output, target[:, c])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
        for p in decoder.parameters():
            p.data.add_(-0.01, p.grad.data)  # 学习率 0.01

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{args.n_epochs}, Loss: {loss.item()}")
