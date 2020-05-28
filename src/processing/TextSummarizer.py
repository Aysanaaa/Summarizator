from abc import abstractmethod

from src.preprocessing.Preprocessing import Preprocessing


class TextSummarizer:

    def __init__(self):
        self.preprocessing = Preprocessing()

    @abstractmethod
    def predict(self,data):
        pass
