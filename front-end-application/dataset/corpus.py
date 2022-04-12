import pandas as pd
from pathlib import Path
from .utils import LookupTable

class Corpus:
    def __init__(self, root):
        self.root = Path(root)
        self.mapping = pd.read_csv(self.root / "annotations" / "gloss_class.csv")

    def create_vocab(self, split):
        df = self.load_data_frame(split)
        sentences = df["annotation"].to_list()
        return LookupTable(
            [gloss for sentence in sentences for gloss in sentence],
            allow_unk=True,
            mapping=self.mapping
        )

class SSLCorpus(Corpus):

    def load_data_frame(self, split):
        """Load corpus."""
        annotations = self.root / "annotations" / f"{split}.csv"
        gloss_classes = self.root / "annotations" / "class.csv"

        df_annotations = pd.read_csv(annotations)
        df_labels = pd.read_csv(gloss_classes)
        df = pd.merge(df_annotations, df_labels[['class','gloss_labels']], on='class', how='inner')
        df.rename(columns = {'gloss_labels':'annotation'}, inplace = True)
        df["annotation"] = df["annotation"].str.split(',')
        df["folder"] = split + "/" + df["folder"]

        return df