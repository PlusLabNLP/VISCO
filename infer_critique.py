import argparse
import json
import os

from evaluate import evaluate_critique
from utils import infer, launch_locally


def format_prompt(prompt_base, item):
    reasoning = "\n".join(["{:d}. {:s}".format(i + 1, x) for i, x in enumerate(item['response']['reasoning'])])
    prompt = prompt_base.replace("{{{QUESTION}}}", item['question']). \
        replace("{{{ANSWER}}}", str(item['response']['answer'])).replace("{{{REASONING}}}", reasoning)

    prompt_lines = prompt.splitlines()
    final_prompt_lines = []
    for line in prompt_lines:
        if '{{{REPEAT_BY_N_STEP}}}' in line:
            for i in range(len(item['response']['reasoning'])):
                final_prompt_lines.append(line.replace('{{{REPEAT_BY_N_STEP}}}', str(i + 1)))
        else:
            final_prompt_lines.append(line)
    prompt = "\n".join(final_prompt_lines)

    return prompt


def format_response(response, n_steps):
    if isinstance(response, list) or isinstance(response, tuple):
        response_history = response[:-1]
        response_orig = response = response[-1]
    else:
        response_history = None
        response_orig = response
    format_error = False

    if not isinstance(response, str):
        ret = {'response': response, 'formatted': None, 'format_error': True}
        if response_history is not None:
            ret['response_history'] = response_history
        return ret

    response = response.replace('\_', '_').replace('\\', '\\\\')
    try:
        response = response.split('```json')[-1].split('```')[0]
        response = json.loads(response)
        assert isinstance(response, dict)
    except:
        try:
            response = '{' + "{".join(response.split('{')[1:])
            response = "}".join(response.split('}')[:-1]) + '}'
            response = json.loads(response)
            assert isinstance(response, dict)
        except:
            response = {}
            format_error = True

    def to_true_or_false(x):
        if isinstance(x, bool):
            return x
        elif isinstance(x, str):
            x = x.lower().strip()
            if x == 'correct' or x == 'yes' or x == 'true':
                return True
            elif x == 'incorrect' or x == 'no' or x == 'false':
                return False
        return None

    def process_dict_key(x):
        if isinstance(x, dict):
            return {k.strip().lower(): process_dict_key(v) for k, v in x.items()}
        return x

    response = process_dict_key(response)
    formatted = {}

    formatted['answer_correctness'] = None
    if 'answer_correctness' in response:
        if isinstance(response['answer_correctness'], dict):
            if 'correctness' in response['answer_correctness']:
                formatted['answer_correctness'] = to_true_or_false(response['answer_correctness']['correctness'])
        else:
            formatted['answer_correctness'] = to_true_or_false(response['answer_correctness'])
    if formatted['answer_correctness'] is None:
        format_error = True

    formatted['reasoning_correctness'] = [None for _ in range(n_steps)]
    formatted['reasoning_critic'] = [None for _ in range(n_steps)]
    for i in range(n_steps):
        if 'step_{:d}'.format(i + 1) in response:
            step_response = response['step_{:d}'.format(i + 1)]
            if isinstance(step_response, dict) and 'correctness' in step_response:
                formatted['reasoning_correctness'][i] = to_true_or_false(step_response['correctness'])
                if 'explanation' in step_response:
                    formatted['reasoning_critic'][i] = str(step_response['explanation'])
            if formatted['reasoning_correctness'][i] is None or formatted['reasoning_critic'][i] is None:
                format_error = True

    ret = {'response': response_orig, 'formatted': formatted, 'format_error': format_error}
    if response_history is not None:
        ret['response_history'] = response_history
    return ret


def main(args):
    with open(args.input) as f:
        data = [json.loads(line) for line in f]

    prompt_fname = os.path.join(os.path.dirname(__file__), 'prompts/critique.txt')
    with open(prompt_fname) as f:
        PROMPT = f.read()

    prompts = [format_prompt(PROMPT, item) for item in data]
    print("\n--- Example prompt")
    print(prompts[0])
    images = [item['image'] for item in data]
    responses = infer(prompts, images, args)

    responses = [format_response(response, len(item['response']['reasoning']))
                 for response, item in zip(responses, data)]
    for i in range(5):
        print("\n--- Example parse:\n")
        print(responses[i]['response'])
        print("\n--->\n")
        print(json.dumps(responses[i]['formatted'], indent=2))

    if args.output is not None:
        print("Save outputs to", args.output)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            for r in responses:
                f.write(json.dumps(r) + '\n')

    evaluate_critique(data, responses)


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
