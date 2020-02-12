import librosa
import time
import torch
from model import *
import numpy as np
import json
import random
import zipfile


# 键形列表
jx_list = [i for i in range(1, 16)]


def get_audio_features(x, sr, bpm, position, offset):
    one_beat = 60 / bpm
    beat = position * one_beat / 4 - offset / 1000

    start = beat
    end = start + one_beat / 8

    end2 = start + one_beat / 4
    if start < 0:
        start = 0

    #     print(start, end)
    start_index = int(sr * start)
    end_index = int(sr * end)

    #     start_index2 = int(sr * start2)
    end_index2 = int(sr * end2)
    check_end = end_index2 + int(2*one_beat*sr)
    if check_end > len(x):
        return []

    features = []
    mfcc1 = librosa.feature.mfcc(y=x[start_index:end_index], sr=sr, n_mfcc=32)
    mfcc2 = librosa.feature.mfcc(y=x[end_index:end_index2], sr=sr, n_mfcc=32)

    features += [float(np.mean(e)) for e in mfcc1]
    features += [float(np.mean(e)) for e in mfcc2]

    return features


# 歌曲基本信息
BPM = 200
offset = 0.7024
song_name = "china-p"
artist = "徐梦圆"
audio_file = "china-p.mp3"
# BPM = 166
# offset = 107.25
# song_name = "万神纪"
# artist = "肥皂菌版"
# audio_file = "wanshenji.mp3"
# x, sr = librosa.load(audio_file, sr=20000)
# print(x.shape)
X = []
START_POSITION = 4
position = START_POSITION

# with open("X.json", "r") as f:
#     X = json.load(f)

feature = get_audio_features(x, sr, BPM, position, offset)
while len(feature) != 0:
    # print(position)
    X.append(feature)
    position += 1
    feature = get_audio_features(x, sr, BPM, position, offset)

# with open("X2.json", "w") as f:
#     json.dump(X, f)

print(len(X))


mc_data = {
    "meta": {
        "creator": "nladuo",
        "version": "Generated with Artificial Intelligence",
        "mode": 0,
        "time": int(time.time()),
        "song": {
            "title": song_name,
            "artist": artist,
        },
        "mode_ext": {
            "column": 4
        }
    },
    "time": [
        {
            "beat": [0, 0, 1],
            "bpm": BPM
        },
    ],
    "extra": {
        "test": {
            "divide": 4,
            "speed": 100,
            "save": 0,
            "lock": 0,
            "edit_mode": 0
        }
    }
}

last_note = {
    "beat": [0, 0, 1],
    "sound": audio_file,
    "vol": 100,
    "offset": offset,
    "type": 1
}


encoder_ln1 = torch.load("checkpoints/encoder_ln1.pth").to(device)
encoder_ln2 = torch.load("checkpoints/encoder_ln2.pth").to(device)
ln_cls_layer = torch.load("checkpoints/ln_cls_layer.pth").to(device)

ln_encoder = torch.load("checkpoints/ln_encoder.pth").to(device)
ln_decoder = torch.load("checkpoints/ln_decoder.pth").to(device)


encoder_beat1 = torch.load("checkpoints/encoder_beat1.pth").to(device)
encoder_beat2 = torch.load("checkpoints/encoder_beat2.pth").to(device)
beat_cls_layer = torch.load("checkpoints/beat_cls_layer.pth").to(device)

beat_encoder = torch.load("checkpoints/beat_encoder.pth").to(device)
beat_decoder = torch.load("checkpoints/beat_decoder.pth").to(device)


max_length = len(X)


def get_decode_result(X, encoder1, encoder2, cls_layer, encoder, decoder, filter_prob=0.5, topn=3):
    global max_length
    with torch.no_grad():
        X = torch.from_numpy(np.array(X)).to(device).float()
        # max_length = 100
        encoder10_outputs = torch.zeros(max_length, encoder1.hidden_size, device=device)
        encoder20_outputs = torch.zeros(max_length, encoder2.hidden_size, device=device)

        encoder_hidden = encoder1.initHidden()
        for ei in range(max_length):
            # print(x[ei])
            encoder_output, encoder_hidden = encoder1(
                X[ei], encoder_hidden)
            encoder10_outputs[ei] = encoder_output[0, 0]

        encoder_hidden = encoder2.initHidden()
        for ei in range(max_length):
            # print(x[ei])
            encoder_output, encoder_hidden = encoder2(
                X[max_length - ei - 1], encoder_hidden)
            encoder20_outputs[max_length - ei - 1] = encoder_output[0, 0]

        encoder_outputs = torch.cat([encoder10_outputs, encoder20_outputs], dim=1)

        # 判断是否有键
        cls_result = []
        X2 = []
        for di in range(max_length):
            cls_input = encoder_outputs[di]
            cls_output = cls_layer(cls_input)
            # _, topi = torch.topk(cls_output, 1)
            output = np.exp(cls_output.numpy())
            one_prob = output[0][1]
            if one_prob > filter_prob:
                cls_result.append(di+START_POSITION)
                X2.append(X[di])


        # 判断键形
        now_position = 0
        decoder_len = len(X2)
        decoder_result = {}
        count = 0
        label = random.choice(jx_list)
        while now_position < decoder_len:  # 防止rnn丢失记忆。这里分成40个一组
            m_counter = 0
            encoder_hidden = encoder.initHidden()
            start_position = now_position
            while True:
                _, encoder_hidden = encoder(
                    X2[now_position], encoder_hidden)

                m_counter += 1
                if m_counter > 40:
                    break
                now_position += 1
                if now_position == decoder_len:
                    break

            decoder_input = torch.tensor([[label]], device=device)
            decoder_hidden = encoder_hidden
            for fi in range(start_position, now_position):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                _, topi = torch.topk(decoder_output, topn)
                label = random.choice(topi[0])
                decoder_input = torch.tensor([[label]], device=device)

                if label != 0:
                    count += 1
                    print("decoder output:", count, cls_result[fi], label)
                    decoder_result[cls_result[fi]] = int(label)
        print(count)
        return decoder_result


ln_decoder_result = get_decode_result(X, encoder_ln1, encoder_ln2,
                                      ln_cls_layer, ln_encoder, ln_decoder, filter_prob=0.6, topn=2)
beat_decoder_result = get_decode_result(X, encoder_beat1, encoder_beat2, beat_cls_layer,
                                        beat_encoder, beat_decoder, filter_prob=0.45, topn=2)


col = {
    0: {},
    1: {},
    2: {},
    3: {},
}

for k in beat_decoder_result.keys():
    jx = beat_decoder_result[k]
    col[0][k] = jx % 2
    col[1][k] = int(jx / 2) % 2
    col[2][k] = int(jx / 4) % 2
    col[3][k] = int(jx / 8) % 2

for k in ln_decoder_result.keys():
    jx = ln_decoder_result[k]

    col[0][k] = jx % 2 * 2
    col[1][k] = int(jx / 2) % 2 * 2
    col[2][k] = int(jx / 4) % 2 * 2
    col[3][k] = int(jx / 8) % 2 * 2


notes = []

for i in range(START_POSITION, START_POSITION+max_length):
    beat = int(i / 4)
    sub_beat = i % 4

    for k in col.keys():
        if i in col[k]:
            if col[k][i] == 1:
                notes.append({
                    "beat": [beat, sub_beat, 4],
                    "column": k
                })
            elif col[k][i] == 2:
                # 如果是开始
                if ((i-1) in col[k]) and (col[k][i-1] == 2):
                    pass
                elif ((i-2) in col[k]) and (col[k][i-2] == 2):
                    pass
                else:
                    end_position = i + 1
                    end_position2 = i + 2
                    while ((end_position in col[k]) and (col[k][end_position] == 2)):#  or \
                            # ((end_position2 in col[k]) and (col[k][end_position2] == 2)):
                        end_position += 1
                        end_position2 += 1

                    end_position -= 1
                    if end_position == i:
                        notes.append({
                            "beat": [beat, sub_beat, 4],
                            "column": k
                        })
                    else:
                        end_beat = int(end_position / 4)
                        end_sub_beat = end_position % 4
                        notes.append({
                            "beat": [beat, sub_beat, 4],
                            "endbeat": [end_beat, end_sub_beat, 4],
                            "column": k
                        })
notes.append(last_note)

mc_data["note"] = notes

with open("generated_with_ai.mc", "w") as f:
    json.dump(mc_data, f)

with zipfile.ZipFile("generated-with-ai.mcz", 'w') as z:
    z.write("generated_with_ai.mc")
    z.write(audio_file)
