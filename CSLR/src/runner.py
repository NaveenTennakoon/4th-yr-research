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
from jiwer import ReduceToSingleSentence, compute_measures

from .model import Model


class Runner(torchzq.LegacyRunner):
    def __init__(
        self,
        # dataset
        data_root: Path = "data",
        p_drop: float = 0.5,
        # augmentation
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
            random_flip=self.training,
            random_jitter=self.training,
            crop_size=args.crop_size,
            base_size=args.base_size,
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

        # SINGLE INPUT
        x, y = batch["video"], batch["label"]
        for i in range(len(x)):
            x[i] = x[i].to(args.device)
            y[i] = y[i].to(args.device)
        batch["video"] = x

        # MULTIPLE INPUTS
        # x1, x2, y = batch["f_frames"], batch["l_frames"], batch["label"]
        # for i in range(len(x1)):
            # x1[i] = x1[i].to(args.device)
            # x2[i] = x2[i].to(args.device)
            # y[i] = y[i].to(args.device)
        # batch["f_frames"] = x1
        # batch["l_frames"] = x2

        batch["label"] = y
        self.batch = batch
        return batch

    def compute_loss(self, batch):
        # SINGLE INPUT
        return self.model.compute_loss(batch["video"], batch["label"])
        # MULTIPLE INPUTS
        # return self.model.compute_loss(batch["f_frames"], batch["l_frames"], batch["label"])

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
            
            # SINGLE INPUT
            for batch in tqdm.tqdm(self.data_loader):
                batch = self.prepare_batch(batch)
                video = batch["video"]
                prob += [lpi.exp().cpu().numpy() for lpi in self.model(video)]

            # MULTIPLE INPUTS
            # for batch in tqdm.tqdm(self.data_loader):
            #     batch = self.prepare_batch(batch)
            #     f_frames = batch["f_frames"]
            #     l_frames = batch["l_frames"]
            #     prob += [lpi.exp().cpu().numpy() for lpi in self.model(f_frames, l_frames)]

            np.savez_compressed(prob_path, prob=prob)

        hyp = self.model.decode(
            prob,
            args.beam_width,
            args.prune,
            args.lm,
            args.nj,
        )

        # WER calculation steps
        ground_truths = []        
        for sentence in self.dataset.data_frame["annotation"]:
            label = []
            for word in sentence:
                label.append(self.vocab.mapping[word])
            ground_truths.append(label)

        gt = []
        pt = []
        for tl, pl in zip(ground_truths, hyp):
            tl = [str(x) for x in tl]
            tl = ReduceToSingleSentence()(tl)
            pl = [str(x) for x in pl]            
            pl = ReduceToSingleSentence()(pl)
            if pl == []:
                pl = ['']
            gt += tl
            pt += pl

        measures = compute_measures(gt, pt)
        measures['wer'] = round(measures['wer'] * 100, 2)
        measures['mer'] = round(measures['mer'] * 100, 2)
        measures['wil'] = round(measures['wil'] * 100, 2)
        measures['wip'] = round(measures['wip'] * 100, 2)
        total_gt_words = measures['hits'] + measures['substitutions'] + measures['deletions']
        hit_percentage = round((measures['hits'] / total_gt_words) * 100, 2)
        sub_percentage = round((measures['substitutions'] / total_gt_words) * 100, 2)
        del_percentage = round((measures['deletions'] / total_gt_words) * 100, 2)
        ins_percentage = round((measures['insertions'] / total_gt_words) * 100, 2)

        print("\n OVERALL MEASURES \n\n", \
            f"Word Error rate (WER) : {measures['wer']} \n", \
            f"Match Error Rate (MER) : {measures['mer']} \n", \
            f"Word Information Lost (WIL) : {measures['wil']} \n", \
            f"Word Information Preserved (WIP) : {measures['wip']} \n", \
            f"Hits : {measures['hits']} ({hit_percentage}%) \n", \
            f"Substitutions : {measures['substitutions']} ({sub_percentage}%) \n", \
            f"Deletions : {measures['deletions']} ({del_percentage}%) \n", \
            f"Insertions : {measures['insertions']} ({ins_percentage}%)", \
        )

    @staticmethod
    def write_txt(path, content):
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            f.write(content)


if __name__ == "__main__":
    torchzq.start(Runner)
