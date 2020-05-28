from transformers import BertTokenizer, TFBertForNextSentencePrediction
import tensorflow as tf
import tensorflow.keras
import numpy as np

pretrained_weights = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
nsp_model = TFBertForNextSentencePrediction.from_pretrained(pretrained_weights)


def reorder_sentences(sentences: list):
    ordering = []
    correlation_matrix = create_correlation_matrix(sentences)
    hint = None
    while correlation_matrix.any():
        if hint == None:
            ind = np.unravel_index(np.argmax(correlation_matrix, axis=None), correlation_matrix.shape)
        else:
            ind = np.unravel_index(np.argmax(correlation_matrix[hint,:], axis=None), correlation_matrix[hint,:].shape)
            ind = (hint, ind[0])
        hint = ind[1]
        correlation_matrix[ind[0], :] = 0
        correlation_matrix[:, ind[0]] = 0
        ordering.append(ind[0])

    reordered_sentences = [sentences[idx] for idx in ordering]
    return reordered_sentences

def create_correlation_matrix(sentences: list):
    num_sentences = len(sentences)
    correlation_matrix = np.empty(shape=(num_sentences, num_sentences), dtype=float, order='C')
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            correlation_matrix[i][j] = predict_next_sentence_prob(s1, s2)
    return correlation_matrix

def predict_next_sentence_prob(sent1, sent2):
    # encode the two sequences. Particularly, make clear that they must be
    # encoded as "one" input to the model by using 'seq_B' as the 'text_pair'
    # NOTE how the token_type_ids are 0 for all tokens in seq_A and 1 for seq_B,
    # this way the model knows which token belongs to which sequence
    encoded = tokenizer.encode_plus(sent1, text_pair=sent2)
    encoded["input_ids"] = tf.constant(encoded["input_ids"])[None, :]
    encoded["token_type_ids"] = tf.constant(encoded["token_type_ids"])[None, :]
    encoded["attention_mask"] = tf.constant(encoded["attention_mask"])[None, :]

    # a model's output is a tuple, we only need the output tensor containing
    # the relationships which is the first item in the tuple
    outputs = nsp_model(encoded)
    seq_relationship_scores = outputs[0]

    # we need softmax to convert the logits into probabilities
    # index 0: sequence B is a continuation of sequence A
    # index 1: sequence B is a random sequence
    probs = tf.keras.activations.softmax(seq_relationship_scores, axis=-1)
    return probs.numpy()[0][0]