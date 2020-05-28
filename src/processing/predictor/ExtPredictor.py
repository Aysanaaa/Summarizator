import torch
from src.models.ExtSummarizer import ExtSummarizer
from src.processing.predictor.Predictor import Predictor
from src.processing.traslator.ExtractiveTraslator import build_trainer


class ExtPredictor(Predictor):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.model = torch.load('./resources/models/summarizer_extractive.pt',
                                map_location=self.device)
        self.model_ext = ExtSummarizer(self.args, self.device, self.model)
        self.model_ext.eval()
        self.trainer = self.build_predictor()

    def build_predictor(self):
        trainer = build_trainer(self.args, -1, self.model_ext, None)
        return trainer

    def predict(self,data_iterator):
        with torch.no_grad():
            return self.trainer.translate_batch(data_iterator)
