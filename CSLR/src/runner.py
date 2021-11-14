import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torchzq
import textwrap
import imageio
from torchzq.parsing import boolean, custom
from pathlib import Path
from einops import rearrange
from dataset_utils import SSLVideoTextDataset
from jiwer import wer, ReduceToSingleSentence

from .fusion_model import Model


class Runner(torchzq.LegacyRunner):
    def __init__(
        self,
        # dataset
        data_root: Path = "data",
        p_drop: float = 0.5,
        # augmentation
        lip_base_size: custom(type=int, nargs=2) = [64, 64],
        base_size: custom(type=int, nargs=2) = [256, 256],
        crop_size: custom(type=int, nargs=2) = [224, 224],
        # model
        max_num_states: int = 5,
        dim: int = 512,
        rdim: int = 32,
        p_detach: float = 0.75,
        rpe_k: int = 8,
        use_sfl: boolean = True,
        heads: int = 4,
        semantic_layers: int = 2,
        dropout: float = 0.1,
        # loss
        ent_coef: float = 0.01,
        monte_carlo_samples: int = 32,
        # decode
        beam_width: int = 10,
        prune: float = 1e-2,
        # debug
        head: int = None,
        split: str = None,
        # misc
        seed: int = 0,
        use_lm: boolean = False,
        **kwargs,
    ):
        self.update_args(locals(), "self")
        super().__init__(**kwargs)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def create_model(self) -> Model:
        model = self.autofeed(Model, dict(vocab_size=len(self.vocab)))
        print("==> #params:", sum(p.numel() for p in model.parameters()))
        return model

    @property
    def split(self):
        args = self.args
        if args.split is None:
            split = self.mode
        else:
            split = args.split
        if split == "validate":
            split = "test"
        return split

    def create_dataset(self):
        args = self.args

        dataset = SSLVideoTextDataset(
            root=args.data_root,
            split=self.split,
            p_drop=args.p_drop,
            random_drop=self.training,
            random_crop=self.training,
            crop_size=args.crop_size,
            base_size=args.base_size,
            lip_base_size=args.lip_base_size,
        )

        if args.head is not None:
            dataset.data_frame = dataset.data_frame.head(args.head)

        print(textwrap.shorten(f"==> Vocab: {dataset.vocab}", 300))
        print(f"==> Vocab size: {len(dataset.vocab)}")

        self.vocab = dataset.vocab
        args.collate_fn = dataset.collate_fn

        if args.use_lm:
            args.lm = dataset.corpus.create_lm()
        else:
            args.lm = None

        return dataset

    def create_data_loader(self, **kwargs):
        return super().create_data_loader(drop_last=self.training, **kwargs)

    def prepare_batch(self, batch):
        args = self.args
        x1, x2, y = batch["f_frames"], batch["l_frames"], batch["label"]
        # x, y = batch["video"], batch["label"]
        for i in range(len(x1)):
            x1[i] = x1[i].to(args.device)
            x2[i] = x2[i].to(args.device)
            # x[i] = x[i].to(args.device)
            y[i] = y[i].to(args.device)
        batch["f_frames"] = x1
        batch["l_frames"] = x2
        # batch["video"] = x
        batch["label"] = y
        self.batch = batch
        return batch

    def compute_loss(self, batch):
        return self.model.compute_loss(batch["f_frames"], batch["l_frames"], batch["label"])
        # return self.model.compute_loss(batch["video"], batch["label"])

    @property
    def result_dir(self):
        return Path("results", self.name, str(self.model.epoch), self.split)

    @torchzq.command
    @torch.no_grad()
    def test(self):
        args = self.args
        self.switch_mode("test")

        prob_path = self.result_dir / "prob.npz"
        prob_path.parent.mkdir(exist_ok=True, parents=True)

        if prob_path.exists():
            prob = np.load(prob_path, allow_pickle=True)["prob"]
        else:
            prob = []
            for batch in tqdm.tqdm(self.data_loader):
                batch = self.prepare_batch(batch)
                f_frames = batch["f_frames"]
                l_frames = batch["l_frames"]
                # video = batch["video"]
                prob += [lpi.exp().cpu().numpy() for lpi in self.model(f_frames, l_frames)]
                # prob += [lpi.exp().cpu().numpy() for lpi in self.model(video)]
            np.savez_compressed(prob_path, prob=prob)

        hyp = self.model.decode(
            prob,
            args.beam_width,
            args.prune,
            args.lm,
            args.nj,
        )

        ground_truths = []        
        for sentence in self.dataset.data_frame["annotation"]:
            label = []
            for word in sentence:
                label.append(self.vocab.mapping[word])
            ground_truths.append(label)

        gt = []
        pt = []
        # WER = 0
        for tl, pl in zip(ground_truths, hyp):
            tl = [str(x) for x in tl]
            tl = ReduceToSingleSentence()(tl)
            pl = [str(x) for x in pl]            
            pl = ReduceToSingleSentence()(pl)
            if pl == []:
                pl = ['']
            gt += tl
            pt += pl
            # wer_cal = wer(tl, pl) 
            # WER += wer_cal

        WER = wer(gt, pt) * 100
        print(WER)

    @staticmethod
    def write_txt(path, content):
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            f.write(content)


if __name__ == "__main__":
    torchzq.start(Runner)
