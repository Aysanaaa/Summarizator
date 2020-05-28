# -*- coding: ISO-8859-1 -*-
import nltk
from nltk import word_tokenize, sent_tokenize
nltk.download('punkt')


def change_split_parameters(max_tokens_span, overlapping_sentences):
    max_tokens_span += 20
    overlapping_sentences -= 1
    if max_tokens_span > 512:
        max_tokens_span = 512
    if overlapping_sentences < 0:
        overlapping_sentences = 0
    return max_tokens_span, overlapping_sentences


def initialize_text_part(sentence_data_list=()):
    text_part = {
        "num_tokens": 0,
        "num_sentences": 0,
        "text": ''
    }
    text_part = aggregate_text_part(text_part, sentence_data_list)
    return text_part


def aggregate_text_part(text_part, sentence_data_list):
    for sentence_data in sentence_data_list:
        text_part["num_tokens"] += sentence_data["num_tokens"]
        text_part["num_sentences"] += 1
        text_part["text"] += ' ' + sentence_data["text"]
    return text_part


def text_split(text, max_tokens_span=400, overlapping_sentences=1):
    sentences = sent_tokenize(text)
    sentence_data_map = dict()
    for idx, sentence in enumerate(sentences):
        sentence_data_map[idx] = {
            "num_tokens": len(word_tokenize(sentence)),
            "text": sentence,
        }

    text_parts = []
    text_part = initialize_text_part()
    for idx, sentence_data in sentence_data_map.items():
        if text_part["num_tokens"] < max_tokens_span:
            aggregate_text_part(text_part, [sentence_data])
        else:
            text_parts.append(text_part)
            sentence_data_list = []
            if idx > overlapping_sentences:
                for i in range(1, overlapping_sentences):
                    sentence_data_list.append(sentence_data_map[idx - i])
            text_part = initialize_text_part(sentence_data_list)
    return text_parts
