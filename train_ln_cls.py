import json
from torch import optim
import random
from model import *

with open("dataset.json", "r") as f:
    dataset = json.load(f)
X0 = dataset['X2']
Y0 = dataset['Y2']


hidden_size = 128
encoder_f = EncoderRNN(hidden_size).to(device)  # 前向
encoder_b = EncoderRNN(hidden_size).to(device)  # 逆向
cls_layer = OutPutLayer(hidden_size, 2).to(device)

learning_rate = 1e-3
encoder_optimizer = optim.Adam(encoder_f.parameters(), lr=learning_rate)
encoder_optimizer2 = optim.Adam(encoder_b.parameters(), lr=learning_rate)
cls_layer_optimizer = optim.Adam(cls_layer.parameters(), lr=learning_rate)


SOS_token = 0


for epoch in range(20):
    total_loss = 0
    for i in range(1000):
        index = random.randrange(0, len(X0))
        x0 = torch.from_numpy(np.array(X0[index])).to(device).float()
        y0 = torch.from_numpy(np.array(Y0[index])).to(device).long()

        max_length = len(X0[index])

        encoder_optimizer.zero_grad()
        encoder_optimizer2.zero_grad()
        cls_layer_optimizer.zero_grad()

        encoder_f_outputs = torch.zeros(max_length, encoder_f.hidden_size, device=device)
        encoder_b_outputs = torch.zeros(max_length, encoder_b.hidden_size, device=device)
        loss = 0

        encoder_f_hidden = encoder_f.initHidden()
        for ei in range(max_length):
            # print(x[ei])
            encoder_output, encoder_f_hidden = encoder_f(
                x0[ei], encoder_f_hidden)
            encoder_f_outputs[ei] = encoder_output[0, 0]

        encoder_b_hidden = encoder_b.initHidden()
        for ei in range(max_length):
            # print(x[ei])
            encoder_output, encoder_b_hidden = encoder_b(
                x0[max_length-ei-1], encoder_b_hidden)
            encoder_b_outputs[max_length-ei-1] = encoder_output[0, 0]

        encoder_outputs = torch.cat([encoder_f_outputs, encoder_b_outputs], dim=1)

        # 判断是否有键
        for di in range(max_length):
            cls_input = encoder_outputs[di]
            cls_output = cls_layer(cls_input)
            target = y0[di].view(-1)

            loss += F.nll_loss(cls_output, target)

        loss.backward()
        encoder_optimizer.step()
        encoder_optimizer2.step()
        cls_layer_optimizer.step()

        total_loss += loss.item() / max_length
        if i % 20 == 0:
            avg_loss = total_loss / (i+1)
            now_loss = loss.item() / max_length
            print(f"Epoch: {epoch},i: {i},avg_loss:{avg_loss},loss:{now_loss}")

    # save models
    torch.save(encoder_f, f"checkpoints/encoder_ln1.pth")
    torch.save(encoder_b, f"checkpoints/encoder_ln2.pth")
    torch.save(cls_layer, f"checkpoints/ln_cls_layer.pth")
