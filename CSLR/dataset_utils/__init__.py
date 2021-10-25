from .datasets import VideoTextDataset
from .corpora import SSLCorpus

class SSLVideoTextDataset(VideoTextDataset):
    Corpus = SSLCorpus