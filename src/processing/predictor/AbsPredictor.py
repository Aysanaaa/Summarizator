import torch
from models.AbsSummarizer import AbsSummarizer
from preprocessing.tokenizer import BertTokenizer
from processing.predictor.GNMTGlobalScorer import GNMTGlobalScorer
from processing.predictor.Predictor import Predictor
from processing.traslator.AbstractiveTraslator import AbstractiveTranslator


class AbsPredictor(Predictor):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.model = torch.load('./resources/models/summarizer_abstractive.pt',
                                map_location=self.device)
        self.model_abs = AbsSummarizer(self.args, self.device, self.model)
        self.tokenizer = BertTokenizer.from_pretrained(
            './resources/models/summarizer_abstractive.pt', do_lower_case=True,
            cache_dir=self.args.temp_dir)
        self.symbols = {'BOS': self.tokenizer.vocab['[unused0]'],
                        'EOS': self.tokenizer.vocab['[unused1]'],
                        'PAD': self.tokenizer.vocab['[PAD]'],
                        'EOQ': self.tokenizer.vocab['[unused2]']}
        self.translator = self.build_predictor()

    def build_predictor(self):
        scorer = GNMTGlobalScorer(self.args.alpha, length_penalty='wu')
        translator = AbstractiveTranslator(self.args, self.model_abs,
                                           self.tokenizer, self.symbols,
                                           global_scorer=scorer, logger=None)
        return translator

    def predict(self, data_iterator):
        batch_predictions = []
        with torch.no_grad():
            for batch in data_iterator:
                batch_data = self.translator.translate_batch(batch)
                translations = self.translator.from_batch(batch_data)
                for trans in translations:
                    pred, src = trans
                    pred_str = pred.replace('[unused0]', '').replace(
                        '[unused3]', '').replace('[PAD]', '').replace(
                        '[unused1]',
                        '').replace(
                        r' +', ' ').replace(' [unused2] ', '<q>').replace(
                        '[unused2]', '').strip()
                    # print(src)
                    # print(pred_str)
                    batch_predictions.append(pred_str)
        return batch_predictions
