import glob
import os
import sys
sys.path.append("./")
import random

import torch
import pandas as pd
import warnings
warnings.simplefilter('ignore')

from hparam import hparam as hp
from model import FFNet
from dataloader import audio_to_dvector, utters_to_dvectors
from preprocess import get_speakers_dict




def gausian_kernel(di, dj, gamma=1.0):
    diff = di - dj
    norm = torch.norm(diff, p=2, dim=1, keepdim=True)
    return torch.exp(-gamma * norm)

def main(gender="female", user_audio=""):
    movel_path = os.path.join(os.path.dirname(__file__),
                              hp.train.checkpoint_dir,
                              f"final_epoch_{hp.train.iteration}.model")

    device = torch.device(hp.device)

    net = FFNet().to(device)
    net.load_state_dict(torch.load(movel_path))
    net.eval()

    d_vectors, speakers = [], []
    user_name = "actors"
    search_path = os.path.join(os.path.dirname(__file__), hp.actors_data, '*.wav')
    for wavfile in glob.glob(search_path):
        speakers.append(os.path.basename(wavfile).split(".")[0])
        d_vectors.append(audio_to_dvector(wavfile, net, device))

    if user_audio != "":
        user_name = os.path.basename(user_audio).split(".")[0]
        speakers.append(user_name)
        d_vectors.append(audio_to_dvector(user_audio, net, device))

    Ns = len(d_vectors)
    utter = torch.stack(d_vectors)
    ks = [gausian_kernel(utter, utter[i]) for i in range(Ns)]
    gram_matrix = torch.cat(ks, dim=1)
    gram_matrix = gram_matrix.cpu().detach().numpy()

    simmat = pd.DataFrame(data=gram_matrix, index=speakers, columns=None)

    output_dir = os.path.join(os.path.dirname(__file__), hp.test.simmat_dir)
    os.makedirs(output_dir, exist_ok=True)
    fn = os.path.join(output_dir, f"simmat_{user_name}.csv")
    simmat.to_csv(fn, index=True, header=False)
    print("output: ", fn)



if __name__ == "__main__":
    if len(sys.argv) > 2:
        user_audio = sys.argv[1]
        gender = sys.argv[2]
    else:
        user_audio = ""
        gender = "female"

    main(gender=gender, user_audio=user_audio)