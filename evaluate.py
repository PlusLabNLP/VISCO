import argparse
import copy
import json
import os

import numpy as np
import tabulate

from gpt_evaluate import _calc_gpt_metrics
from src_evaluation.evaluate import evaluate


def _f1(gt, pred):
    assert len(gt) == len(pred)
    tp = sum(a is False and b is False for a, b in zip(gt, pred))
    gt_pos = gt.count(False)
    pred_pos = pred.count(False)
    if tp == 0:
        return 0
    else:
        p = tp / pred_pos
        r = tp / gt_pos
        return 2 / (1 / p + 1 / r)


def _evaluate_critic(data, responses, gpt_responses):
    gt_ans = []
    pred_ans = []
    gt_th = []
    pred_th = []
    refs_ex = []
    sys_ex = []
    assert len(data) == len(responses)
    for i in range(len(data)):
        assert len(data[i]['reasoning_critic']) == len(responses[i]['formatted']['reasoning_critic'])

        gt_ans.append(data[i]['answer_correctness'])
        pred_ans.append(responses[i]['formatted']['answer_correctness'])
        gt_th += data[i]['reasoning_correctness']
        pred_th += responses[i]['formatted']['reasoning_correctness']
        refs_ex += data[i]['reasoning_critic']
        sys_ex += responses[i]['formatted']['reasoning_critic']

    ret = {
        'Ans. F1': _f1(gt_ans, pred_ans) * 100,
        'Th. F1': _f1(gt_th, pred_th) * 100,
    }
    if gpt_responses is not None:
        ex_f1 = _calc_gpt_metrics(data, responses, gpt_responses)
        viscore = pow(ret['Ans. F1'] * ret['Th. F1'] * ex_f1, 1 / 3)
        ret['Ex. F1'] = ex_f1
        ret['VISCore'] = viscore
    return ret


def evaluate_critique(data, responses, gpt_responses=None, do_print=True):
    if do_print:
        print("Format error: {:d} / {:d}\n".format(sum(r['format_error'] for r in responses), len(responses)))

    # Remove critic for steps predicted as correct: not necessary
    responses = copy.deepcopy(responses)
    for r in responses:
        for i in range(len(r['formatted']['reasoning_correctness'])):
            if r['formatted']['reasoning_correctness'][i]:
                r['formatted']['reasoning_critic'][i] = ''

    metrics = {
        'Total': _evaluate_critic(data, responses, gpt_responses),
        'Reasoning': _evaluate_critic(
            [data[i] for i in range(len(data)) if data[i]['meta_data']['critic_superskill'] == 'Reasoning'],
            [responses[i] for i in range(len(data)) if data[i]['meta_data']['critic_superskill'] == 'Reasoning'],
            [gpt_responses[i] for i in range(len(data)) if data[i]['meta_data']['critic_superskill'] == 'Reasoning']
            if gpt_responses is not None else None,
        ),
        'Perception': _evaluate_critic(
            [data[i] for i in range(len(data)) if data[i]['meta_data']['critic_superskill'] == 'Perception'],
            [responses[i] for i in range(len(data)) if data[i]['meta_data']['critic_superskill'] == 'Perception'],
            [gpt_responses[i] for i in range(len(data)) if data[i]['meta_data']['critic_superskill'] == 'Perception']
            if gpt_responses is not None else None,
        ),
    }

    if do_print:
        KEYS = ['Ans. F1', 'Th. F1', ]
        if gpt_responses is not None:
            KEYS += ['Ex. F1', 'VISCore', ]
        print(tabulate.tabulate([[category, ] + [metrics[category][k] for k in KEYS] for category in metrics],
                                headers=KEYS, floatfmt=[None, ] + ['.2f', ] * len(KEYS)))

    return metrics


def evaluate_correction(data, responses, do_print=True):
    accuracy_pre = []
    accuracy_post = []
    for i in range(len(data)):
        accuracy_pre.append(not data[i]['id'].startswith("test1"))
        accuracy_post.append(evaluate(responses[i]['answer'], data[i]['label'], data[i]['meta_data']))
    accuracy_pre = np.array(accuracy_pre)
    accuracy_post = np.array(accuracy_post)
    correction_score = accuracy_post[~accuracy_pre].mean() - (1 - accuracy_post)[accuracy_pre].mean()
    accuracy_post = np.mean(accuracy_post)
    if do_print:
        print("Accuracy = {:.2f}".format(accuracy_post * 100))
        print("Correction score = {:.2f}".format(correction_score * 100))
    return {'accuracy': accuracy_post, 'correction_score': correction_score}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--task', default='critique', choices=['critique', 'correction', ])
    parser.add_argument('--input', default='test.jsonl')
    args = parser.parse_args()

    with open(args.input) as f:
        data = [json.loads(line) for line in f]
    with open(args.output) as f:
        responses = [json.loads(line) for line in f]

    if args.task == 'critique':
        gpt_responses = None
        if os.path.exists(args.output + '.gpt_evaluate_cache'):
            with open(args.output + '.gpt_evaluate_cache') as f:
                gpt_responses = [json.loads(line) for line in f]
        evaluate_critique(data, responses, gpt_responses)
    else:
        assert args.task == 'correction'
        evaluate_correction(data, responses)
