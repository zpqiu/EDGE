import os
import json
import torch

from fairseq.data import Dictionary
from fairseq.tasks import FairseqTask, register_task

from .dg_dataset import DGDataset


@register_task("dg_task")
class DGTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument("data", metavar="FILE",
                            help="file prefix for data")
        parser.add_argument("--max_positions", default=1024, type=int,
                            help="max input length")
        parser.add_argument("--max_tgt_positions", default=10, type=int,
                            help="max input length")
        parser.add_argument("--max_q_positions", default=10, type=int,
                            help="max input length")
        parser.add_argument("--max_ans_positions", default=10, type=int,
                            help="max input length")
        parser.add_argument("--max_state_positions", default=10, type=int,
                            help="max input length")

    @classmethod
    def setup_task(cls, args, **kwargs):
        input_vocab = Dictionary.load(os.path.join(args.data, "vocab.txt"))

        print("| [input] dictionary: {} types".format(len(input_vocab)))

        return DGTask(args, input_vocab)

    def __init__(self, args, input_vocab):
        super().__init__(args)
        self.args = args
        self.input_vocab = input_vocab

    def load_dataset(self, split, combine=False, **kwargs):
        prefix = os.path.join(self.args.data, "race_{}.json".format(split))

        sentences, lengths = [], []
        labels, label_lengths = [], []
        qs, q_lengths = [], []
        ans, ans_lengths = [], []
        with open(prefix, encoding="utf8") as file:
            for line in file:
                j = json.loads(line)
                article, distractor, question = self.truncate_seq(j["article"],
                                                                  self.args.max_positions - 1), \
                                                self.truncate_seq(j["distractor"],
                                                                  self.args.max_tgt_positions - 1,
                                                                  cut_left=False), \
                                                self.truncate_seq(j["question"],
                                                                  self.args.max_q_positions - 1)
                answer = self.truncate_seq(j["answer_text"], self.args.max_ans_positions-1)

                sentence = " ".join(article)
                label = " ".join(distractor)
                question = " ".join(question)
                answer = " ".join(answer)

                tokens = self.input_vocab.encode_line(
                    sentence, add_if_not_exist=False,
                ).long()

                sentences.append(tokens)
                lengths.append(tokens.numel())

                tokens = self.input_vocab.encode_line(
                    label, add_if_not_exist=False,
                ).long()

                labels.append(tokens)
                label_lengths.append(tokens.numel())

                tokens = self.input_vocab.encode_line(
                    question, add_if_not_exist=False,
                ).long()

                qs.append(tokens)
                q_lengths.append(tokens.numel())

                tokens = self.input_vocab.encode_line(
                    answer, add_if_not_exist=False,
                ).long()

                ans.append(tokens)
                ans_lengths.append(tokens.numel())

        assert len(sentences) == len(labels)
        print("| {} {} {} examples".format(self.args.data, split, len(sentences)))

        self.datasets[split] = DGDataset(
            src=sentences,
            src_sizes=lengths,
            src_dict=self.input_vocab,
            tgt=labels,
            tgt_sizes=label_lengths,
            q=qs,
            q_sizes=q_lengths,
            ans=ans,
            ans_sizes=ans_lengths,
            left_pad_source=False,
            max_source_positions=self.args.max_positions,
            max_target_positions=self.args.max_tgt_positions,
            max_q_poisitions=self.args.max_q_positions,
            input_feeding=True
        )

    def truncate_seq(self, seq, max_len, cut_left=True):
        if len(seq) <= max_len:
            return seq
        if cut_left:
            return seq[-max_len:]
        return seq[:max_len]

    def max_positions(self):
        return self.args.max_positions, self.args.max_tgt_positions

    @property
    def source_dictionary(self):
        return self.input_vocab

    @property
    def target_dictionary(self):
        return self.input_vocab

