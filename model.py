import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("cpu")

Feature_DIM = 64


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(Feature_DIM, hidden_size)

    def forward(self, input_, hidden):
        input_ = input_.view(1, 1, -1)
        output, hidden = self.gru(input_, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        hidden_size = hidden_size

        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding = embedding
        self.gru = nn.GRU(50, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_, hidden):
        output = self.embedding(input_).view(1, 1, -1)
        output = self.dropout(output)

        # output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


class OutPutLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(OutPutLayer, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_):
        input_ = input_.view(1, -1)
        input_ = self.dropout(input_)
        # output = F.relu(input_)
        output = F.log_softmax(self.out(input_), dim=1)
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def load_word_embeddings(file_name, dim):
    term_ids = {}
    we_matrix = []  # a term_num * dim matrix for word embeddings
    term_ids['NULL'] = 0
    term_by_id = ['NULL']
    we_matrix.append([0] * dim)
    term_num = 1
    with open(file_name) as FileObj:
        for line in FileObj:
            line = line.split()
            term_ids[line[0].strip()] = term_num
            term_by_id.append(line[0].strip())
            norm = 1
            we_matrix.append([float(i) / norm for i in line[-50:]])
            term_num += 1
    return term_ids, term_by_id, we_matrix


def get_glove_embedding(classes, filename):
    print("loading glove embedding.....")
    term_to_id, id_to_term, we_matrix = load_word_embeddings(f"glove/{filename}", 50)
    embedding_matrix = np.random.rand(classes, 50)
    for i in range(81):
        if str(i) in term_to_id:
            tid = term_to_id[str(i)]
            embedding_matrix[i] = we_matrix[tid]
    print("embedding loaded.")
    return torch.FloatTensor(embedding_matrix)


if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

