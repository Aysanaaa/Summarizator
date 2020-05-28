from abc import abstractmethod
from argparse import Namespace

args = Namespace(accum_count=1, alpha=0.95, beam_size=5, block_trigram=True,
                 dec_dropout=0.2, dec_ff_size=2048, dec_heads=8,
                 dec_hidden_size=768, dec_layers=6, enc_dropout=0.2,
                 enc_ff_size=512, enc_hidden_size=512, enc_layers=6,
                 encoder='bert', ext_dropout=0.2, ext_ff_size=2048, ext_heads=8,
                 ext_hidden_size=768, ext_layers=2, finetune_bert=True,
                 generator_shard_size=32, gpu_ranks=[0], label_smoothing=0.1,
                 large=False, max_length=200, max_pos=512, max_tgt_len=140,
                 min_length=50, temp_dir='./resources/models',
                 use_interval=True, world_size=1)


class Predictor:

    def __init__(self):
        self.args = args

    @abstractmethod
    def build_predictor(self):
        pass

    @abstractmethod
    def predict(self, data_iterator):
        pass
