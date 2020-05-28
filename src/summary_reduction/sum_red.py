import numpy as np
#import seaborn as sns
import spacy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
from tensorflow import keras
import tf_sentencepiece
import ssl
#import urllib
#import certifi
ssl._create_default_https_context = ssl._create_unverified_context
#link = urllib.request.urlopen(module_url)
#%matplotlib inline
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/1'
def symmary_reduction(text1):

    doc = [text1]
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        xling_embed = hub.Module(module_url)
        embedded_text = xling_embed(text_input)
        init_options = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    g.finalize()

    session = tf.Session(graph=g)
    session.run(init_options)

    #ranker = 'EmbedRank++'
    rank_fn = embedrankpp

    tokenizer = spacy.load('en_core_web_sm')
    #tokenizer = spacy.load('en')

    sents = [str(s).replace('\n', '') for s in tokenizer(''.join(doc)).sents]
    key_size = len(sents)/3
    # Embedding
    doc_emb = session.run(embedded_text, feed_dict={text_input: doc})
    sent_embs = session.run(embedded_text, feed_dict={text_input: sents})

    # Ranking
    keys = rank_fn(doc_emb, sent_embs, key_size)

    summary = []
    for i, s in enumerate(sents):
        if i in keys:
            summary.append(s)
        else:
            pass
    finalsum = ''.join(summary)
    return finalsum

def ncossim(embs_1, embs_2, axis=0):
    sims = np.inner(embs_1, embs_2)
    std = np.std(sims, axis=axis)
    ex = np.mean((sims-np.min(sims, axis=axis))/np.max(sims, axis=axis), axis=axis)
    return 0.5 + (sims-ex)/std


def mmr(doc_emb, cand_embs, key_embs):
    param = 0.5
    scores = param * ncossim(cand_embs, doc_emb, axis=0)
    if key_embs is not None:
        scores -= (1-param) * np.max(ncossim(cand_embs, key_embs), axis=1).reshape(scores.shape[0], -1)
    return scores

def embedrankpp(doc_emb, sent_embs, n_keys):
    assert 0 < n_keys, 'Please `key_size` value set more than 0'
    assert n_keys < len(sent_embs), 'Please `key_size` value set lower than `#sentences`'
    cand_idx = list(range(len(sent_embs)))
    key_idx = []
    while len(key_idx) < n_keys:
        cand_embs = sent_embs[cand_idx]
        key_embs = sent_embs[key_idx] if len(key_idx) > 0 else None
        scores = mmr(doc_emb, cand_embs, key_embs)
        key_idx.append(cand_idx[np.argmax(scores)])
        cand_idx.pop(np.argmax(scores))
    return key_idx
