import torch
import os
import numpy as np
import model.fusion as md
import cv2

from pathlib import Path
from torchvision import transforms
from PIL import Image

from video_utils import load_lip_detector, bodyFrames2LipFrames
from dataset.corpus import SSLCorpus

def load_model():
    model = md.Model(
        vocab_size=41,
        dim=512,
        max_num_states=1,
    )

    while True:
        path = Path('./model/fusion/ckpt.pth')
        try:
            checkpoint = torch.load(path, "cpu")
            model.load_state_dict(checkpoint['model'])
            model.to("cpu")
            model.eval()
            break
        except Exception as e:
            print(e)
            print("Failed to load model, reloading ...")
            os.remove(path)
    return model


class PyModel:
    def __init__(self):
        # load model
        self.model = load_model()
        # lip extraction networks
        self.pnet, self.rnet, self.onet = load_lip_detector()
        # dataset variables
        self.corpus = SSLCorpus('../CSLR/data/')
        self.vocab = self.corpus.create_vocab("test")
        self.ff_transformer = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.CenterCrop([224,224]),
            transforms.ToTensor(),
        ])
        self.lf_transformer = transforms.ToTensor()
        self.p_drop = 0.75
        # prediction results
        self.prediction = None
        # self.lip_extraction_time = None # Note: comment/remove after calculating avg prediction times
        # self.prediction_time = None # Note: comment/remove after calculating avg prediction times

    def sample_indices(self, n):
        p_kept = 1 - self.p_drop
        indices = np.arange(0, n, 1 / p_kept)
        indices = np.round(indices)
        indices = np.clip(indices, 0, n - 1)
        indices = indices.astype(int)
        return indices
    
    def predict(self, captured_frames):
        # start = time.time() # Note: comment/remove after calculating avg prediction times

        # extract lip frames
        lip_frames = bodyFrames2LipFrames(captured_frames, self.pnet, self.rnet, self.onet)
        # lip_extraction_time = time.time() - start # Note: comment/remove after calculating avg prediction times
        # self.lip_extraction_time = lip_extraction_time # Note: comment/remove after calculating avg prediction times

        if type(lip_frames) != type(None):
            # preprocess the images
            f_frames = list(captured_frames)
            indices = self.sample_indices(len(f_frames))
            f_frames = [f_frames[i] for i in indices]
            f_frames_np = np.array(f_frames)
            f_frames = map(Image.fromarray, f_frames)
            f_frames = map(self.ff_transformer, f_frames)
            f_frames = torch.stack(list(f_frames))
            f_frames = list(torch.unsqueeze(f_frames, dim=0))

            l_frames = list(lip_frames)
            l_frames = [l_frames[i] for i in indices]
            l_frames_np = np.array(l_frames)
            l_frames = map(Image.fromarray, l_frames)
            l_frames = map(self.lf_transformer, l_frames)
            l_frames = torch.stack(list(l_frames))
            l_frames = list(torch.unsqueeze(l_frames, dim=0))

            os.makedirs('./frames/', exist_ok=True)
            for nFrame in range(f_frames_np.shape[0]):
                cv2.imwrite('./frames/' + "/ff_frame%04d.jpg" % nFrame, f_frames_np[nFrame,...])
                cv2.imwrite('./frames/' + "/lf_frame%04d.jpg" % nFrame, l_frames_np[nFrame,...])

            # get prediction from model
            with torch.no_grad():
                lpi, _, _ = self.model(f_frames, l_frames)
                prob = [lpi.exp().cpu().numpy() for lpi in lpi]
            hyp = self.model.decode(prob, 10, 0.01)
            hyp = " ".join([self.vocab[i] for i in hyp])
            self.prediction = hyp
            # get prediction time
            # prediction_time = time.time() - start - lip_extraction_time # Note: comment/remove after calculating avg prediction times
            # self.prediction_time = prediction_time # Note: comment/remove after calculating avg prediction times
        
        results = {
            "prediction": self.prediction,
            # "lip_extraction_time": self.lip_extraction_time, # Note: comment/remove after calculating avg prediction times
            # "prediction_time": self.prediction_time # Note: comment/remove after calculating avg prediction times
        }
        return results