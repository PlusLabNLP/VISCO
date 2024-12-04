import argparse
import copy
import json
import os
import random

from evaluate import evaluate_correction
from utils import launch_locally, infer, get_answer_format


def format_prompt(prompt_base, item, critique, critique_setting):
    answer_format = get_answer_format(item)

    C_map = {True: 'correct.', False: 'incorrect.', None: 'unknown.'}

    reasoning = []
    for i, thought in enumerate(item['response']['reasoning']):
        reasoning.append("{:d}. {:s}".format(i + 1, thought))
        if critique_setting.startswith("AS"):
            c = C_map[critique['reasoning_correctness'][i]]
            if critique['reasoning_correctness'][i] is False:
                c = 'incorrect.'
                if critique_setting == 'ASE':
                    c += ' Explanation: ' + ('unknown.' if critique['reasoning_critic'][i] is None
                                             else critique['reasoning_critic'][i])
            reasoning.append("  - Critique: " + c)
    reasoning.append("{:d}. The final answer is: {}".format(
        len(item['response']['reasoning']) + 1, item['response']['answer']))
    reasoning.append("  - Critique: " + C_map[critique['answer_correctness']])
    reasoning = '\n'.join(reasoning)

    if critique_setting.startswith("AS"):
        critique_setting_text = "each reasoning step"
    else:
        critique_setting_text = "the answer"

    prompt = prompt_base.replace("{{{QUESTION}}}", item['question']).replace("{{{ANSWER_FORMAT}}}", answer_format). \
        replace("{{{REASONING}}}", reasoning).replace("{{{CRITIQUE_SETTING}}}", critique_setting_text)
    if critique_setting == 'A':
        prompt = prompt.replace("{{{REASONING}}}", "the answer")
    else:
        prompt = prompt.replace("{{{REASONING}}}", "each reasoning step as well as the final answer")
    return prompt


def format_response(response):
    response_orig = response
    response = response.replace('\_', '_').replace('\\', '\\\\')

    success = False
    try:
        response = json.loads(response.split('```json')[-1].split('```')[0])
        assert isinstance(response, dict)
        success = True
    except:
        pass

    if not success:
        try:
            response = json.loads(response.split('``` json')[-1].split('```')[0])
            assert isinstance(response, dict)
            success = True
        except:
            pass

    if not success:
        try:
            response = json.loads('{' + response.split('{')[-1].split('}')[0] + '}')
            assert isinstance(response, dict)
        except:
            response = {}

    response = {k.lower().strip(): v for k, v in response.items()}
    answer = str(response.get('answer', ''))
    return {'response': response_orig, 'answer': answer}


def main(args):
    with open(args.input) as f:
        data = [json.loads(line) for line in f]
    if args.critique == 'human':
        critique = copy.deepcopy(data)
        for x in critique:
            for i in range(len(x['reasoning_critic'])):
                assert len(x['reasoning_critic'][i]) == 3
                x['reasoning_critic'][i] = random.choice(x['reasoning_critic'][i])
    else:
        with open(args.critique) as f:
            critique = [json.loads(line)['formatted'] for line in f]

    prompt_fname = os.path.join(os.path.dirname(__file__), 'prompts/correction.txt')
    with open(prompt_fname) as f:
        PROMPT = f.read()

    prompts = [format_prompt(PROMPT, item, c, args.critique_setting) for item, c in zip(data, critique)]
    print("\n--- Example prompt")
    print(prompts[0])
    images = [item['image'] for item in data]
    responses = infer(prompts, images, args)
    responses = [format_response(response) for response in responses]

    if args.output is not None:
        print("Save outputs to", args.output)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            for r in responses:
                f.write(json.dumps(r) + '\n')

    evaluate_correction(data, responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model and inference parameters
    parser.add_argument('--model', default="gpt-4o-2024-08-06")  # auto if we're using a locally served model

    # openai api-based
    parser.add_argument('--api_key', default='YOUR_API_KEY')
    parser.add_argument('--base_url', default=None)
    parser.add_argument('--n_proc', default=16, type=int)
    parser.add_argument('--launch_locally', default=None, choices=['lmdeploy', 'vllm', 'sglang'])

    # input output
    parser.add_argument('--critique', default='human')
    parser.add_argument('--critique_setting', default='ASE', choices=['A', 'AS', 'ASE', ])
    parser.add_argument('--input', default='test.jsonl')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if args.launch_locally:
        process, port = launch_locally(args.launch_locally, args.model)
        args.model = 'auto'
        args.base_url = f'http://0.0.0.0:{port}/v1'

    try:
        main(args)
    finally:
        if args.launch_locally:
            process.kill()
