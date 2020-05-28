from argparse import Namespace

from src.preprocessing.BertData import BertData

args = Namespace(lower=True, use_bert_basic_tokenizer=True,
                 min_src_ntokens_per_sent=5, max_src_ntokens_per_sent=200,
                 max_src_nsents=100, max_tgt_ntokens=500)


class BertTransformation:

    def __init__(self):
        self.args = args

    def to_bert(self, json_data):
        bert = BertData(self.args)
        datasets = []
        for d in json_data:
            source, tgt = d['src'], ''

            sent_labels = ""
            if self.args.lower:
                source = [' '.join(s).lower().split() for s in source]
                tgt = [' '.join(s).lower().split() for s in tgt]
            b_data = bert.preprocess(source, tgt, sent_labels,
                                     use_bert_basic_tokenizer=False,
                                     is_test=True)
            if b_data is None:
                continue
            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                           "src_sent_labels": sent_labels, "segs": segments_ids,
                           'clss': cls_ids,
                           'src_txt': src_txt, "tgt_txt": tgt_txt}
            datasets.append(b_data_dict)
        return datasets
