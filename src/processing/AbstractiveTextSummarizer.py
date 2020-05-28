import torch
from src.data.DataLoader import Dataloader
from src.data.DataLoader import dataset_loader
from src.general.Singleton import Singleton
from src.processing.predictor.AbsPredictor import AbsPredictor

from src.processing import TextSummarizer


class AbstractiveTextSummarizer(TextSummarizer, metaclass=Singleton):

    def __init__(self):
        super().__init__()
        self.predictor = AbsPredictor()

    def predict(self, data):
        processed_data = self.preprocessing.preprocessing(data)
        data_iterator = Dataloader(datasets=dataset_loader(processed_data),
                                   batch_size=64, device=torch.device('cpu'),
                                   shuffle=False, is_test=True)
        summaries = self.predictor.predict(data_iterator)
        response = {"summaries": [[s] for s in summaries]}
        return response
