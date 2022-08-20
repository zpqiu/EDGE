# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2019-09-05

从Lab提供的json中建词表
"""
import json
import argparse
from eval.eval import eval
# import rouge


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def parse_generate_results(result_path):
    lines = open(result_path, "r", encoding="utf8")

    rets = dict()
    for line in lines:
        if line[0] != "H":
            continue
        row = line.strip().split("\t")
        sample_id = int(row[0][2:])
        if sample_id not in rets:
            rets[sample_id] = list()
        rets[sample_id].append(row[-1].split())

    hypothesis = {}
    for key, values in rets.items():
        pred1 = values[0]
        pred2, pred3 = None, None
        for pred in values[1:]:
            if jaccard_similarity(pred1, pred) < 0.5:
                if pred2 is None:
                    pred2 = pred
                else:
                    if pred3 is None:
                        if jaccard_similarity(pred2, pred) < 0.5:
                            pred3 = pred
            if pred2 is not None and pred3 is not None:
                break

        if pred2 is None:
            pred2 = values[1]
            if pred3 is None:
                pred3 = values[2]
        else:
            if pred3 is None:
                pred3 = values[1]

        hypothesis[key] = [pred1, pred2, pred3]
    return hypothesis


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test-path", "--test_path", required=True, type=str)
    parser.add_argument("-gen-path", "--gen_path", type=str, default=None)
    args = parser.parse_args()

    gen_dict = parse_generate_results(args.gen_path)

    lines = open(args.test_path, "r", encoding="utf8").readlines()
    test_dict = dict()

    for i, line in enumerate(lines):
        j = json.loads(line)
        test_dict[i] = j['distractor_list']

    hypo1_list, hypo2_list, hypo3_list, test_list = [], [], [], []
    for key in gen_dict:
        hypo1_list.append(" ".join(gen_dict[key][0]))
        hypo2_list.append(" ".join(gen_dict[key][1]))
        hypo3_list.append(" ".join(gen_dict[key][2]))
        test_list.append([" ".join(x) for x in test_dict[key]])

    # for i, hypo_list in enumerate([hypo1_list, hypo2_list, hypo3_list]):
    #     print("*=" * 10, "The", i, "Distractor", "*=" * 10)
    #     for aggregator in ['Avg', 'Best']:
    #         print('Evaluation with {}'.format(aggregator))
    #         apply_avg = aggregator == 'Avg'
    #         apply_best = aggregator == 'Best'

    #         evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
    #                                 max_n=2,
    #                                 limit_length=True,
    #                                 length_limit=100,
    #                                 length_limit_type='words',
    #                                 apply_avg=apply_avg,
    #                                 apply_best=apply_best,
    #                                 alpha=0.5,  # Default F1_score
    #                                 weight_factor=1.2,
    #                                 stemming=True)

    #         scores = evaluator.get_scores(hypo_list, test_list)
    #         for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    #             print(prepare_results(metric, results['p'], results['r'], results['f']))

    _ = eval(gen_dict, test_dict)


if __name__ == "__main__":
    main()
