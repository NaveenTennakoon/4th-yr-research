from .datasets_fusion import VideoTextDataset
from .corpora_fusion import SSLCorpus

class SSLVideoTextDataset(VideoTextDataset):
    Corpus = SSLCorpus