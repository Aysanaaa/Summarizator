from nltk.tokenize import word_tokenize, sent_tokenize

from src.preprocessing.BertTransformation import BertTransformation


class Preprocessing():

    def __init__(self):
        self.bertTransformation = BertTransformation()

    def tokenize(self, stories):
        stories_list = list()
        for story in stories:
            sentences = sent_tokenize(story)
            src = list()
            for sentence in sentences:
                str_tokens = word_tokenize(sentence)
                src.append(str_tokens)
            data = dict()
            data['src'] = src
            stories_list.append(data)
        return stories_list

    def preprocessing(self, stories):
        stories_list = self.tokenize(stories)
        processed_data = self.bertTransformation.to_bert(stories_list)
        return processed_data
