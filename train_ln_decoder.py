import json
from torch import optim
import random
from model import *


with open("dataset.json", "r") as f:
    dataset = json.load(f)
X3 = dataset['X3']
Y3 = dataset['Y3']

# 键形列表
jx_list = []
with open("glove/vocab2.txt", "r") as f:
    for line in f.readlines():
        num = line.split(" ")[0]
        jx_list.append(int(num))


embedding = nn.Embedding.from_pretrained(get_glove_embedding(16, "vectors2.txt"), freeze=False)

hidden_size = 128
encoder = EncoderRNN(hidden_size).to(device)
decoder = DecoderRNN(embedding, hidden_size, 16).to(device)  # 判断键型

learning_rate = 1e-3
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()


for epoch in range(100):
    total_loss = 0
    for i in range(1000):
        index = random.randrange(0, len(X3))
        x1 = torch.from_numpy(np.array(X3[index])).to(device).float()
        y1 = torch.from_numpy(np.array(Y3[index])).to(device).long()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        max_length = len(X3[index])

        loss = 0

        encoder_hidden = encoder.initHidden()
        for ei in range(max_length):
            _, encoder_hidden = encoder(
                x1[ei], encoder_hidden)

        # 判断是键形
        decoder_input = torch.tensor([[random.choice(jx_list)]], device=device)
        decoder_hidden = encoder_hidden
        for di in range(max_length):

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            target = y1[di].view(-1)
            # print(decoder_output)
            # print(target)
            loss += F.nll_loss(decoder_output, target)
            decoder_input = target  # Teacher forcing

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item() / max_length
        if i % 20 == 0:
            avg_loss = total_loss / (i+1)
            now_loss = loss.item() / max_length
            print(f"Epoch: {epoch},i: {i},avg_loss:{avg_loss},loss:{now_loss}")

    # save models
    torch.save(encoder, f"checkpoints/ln_encoder.pth")
    torch.save(decoder, f"checkpoints/ln_decoder.pth")
