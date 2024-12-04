import argparse
import json
import os

import google.generativeai as genai
import tqdm

import utils
from evaluate import evaluate_critique
from infer_critique import format_response
from utils import launch_locally, func, get_pool

prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts/')
PROMPT_PROBLEM_SOLVER = "{{{QUESTION}}}\nThink step by step, and then provide your final answer."
with open(os.path.join(prompt_dir, 'lookback_visual-query.txt')) as f:
    PROMPT_SCHEDULE_VISUAL_QUERY = f.read()
with open(os.path.join(prompt_dir, 'lookback_synthesize.txt')) as f:
    PROMPT_SYNTHESIZE = f.read()


def func_agent(obj):
    item, image = obj

    # problem solver
    ref_answer = func((image, item['question']))

    def format_prompt(prompt, item):
        prompts = []
        for i in range(len(item['response']['reasoning'])):
            reasoning = "\n".join(["{:d}. {:s}".format(j + 1, x)
                                   for j, x in enumerate(item['response']['reasoning'][:i + 1])])
            which_step = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}[i + 1]
            prompts.append(
                prompt.replace("{{{QUESTION}}}", item['question']). \
                    replace("{{{ANSWER}}}", str(item['response']['answer'])).replace("{{{REASONING}}}", reasoning).
                    replace("{{{WHICH_STEP}}}", which_step)
            )
        return prompts

    def extract_verify_questions(text):
        if 'N/A' in text.strip():
            return []

        text = "\n" + text.strip()
        text = "\n1.".join(text.split("\n1.")[1:])
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("{:d}.".format(i + 1)):
                lines[i] = ".".join(line.split('.')[1:])
            lines[i] = lines[i].strip()
        return lines

    # visual verification
    prompt = format_prompt(PROMPT_SCHEDULE_VISUAL_QUERY, item)
    visual_questions = [func((image, p)) for p in prompt]
    visual_questions = [extract_verify_questions(q) for q in visual_questions]
    visual_answers = [[func((image, pp + ' Answer briefly.')) for pp in p] for p in visual_questions]

    def format_prompt_synthesize(prompt_base):
        reasoning = []
        for i, (r, q, a) in enumerate(zip(item['response']['reasoning'], visual_questions, visual_answers)):
            reasoning.append("{:d}. {:s}".format(i + 1, r))
        reasoning = "\n".join(reasoning)

        visual_info = []
        for q, a in zip(visual_questions, visual_answers):
            for qq, aa in zip(q, a):
                visual_info.append("* {} - {}".format(qq, aa))
        visual_info = "\n".join(visual_info)
        if visual_info.strip() == '':
            visual_info = "N/A"

        prompt = prompt_base.replace("{{{QUESTION}}}", item['question']). \
            replace("{{{ANSWER}}}", str(item['response']['answer'])).replace("{{{REASONING}}}", reasoning). \
            replace("{{{REFERNCE_ANSWER}}}", ref_answer).replace("{{{VISUAL_INFO}}}", visual_info)

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

    # synthesize
    prompt = format_prompt_synthesize(PROMPT_SYNTHESIZE)
    ret = func((image, prompt))
    return ret, {
        'ref_answer': ref_answer, 'visual_questions': visual_questions, 'visual_answers': visual_answers,
    }


def infer(data, images):
    utils.args = args
    if args.model == "gemini-1.5-pro":
        genai.configure(api_key=args.api_key)
    responses = []
    assert len(data) == len(images)
    with get_pool(args.n_proc) as p:
        for response, additional_info in tqdm.tqdm(p.imap(func_agent, zip(data, images)), total=len(images)):
            responses.append((response, additional_info))
            if len(responses) <= 5:
                print("\n\n------------------------- Example output:", len(responses))
                print(responses[-1][0])
                print("\n--- Additional info:")
                print(json.dumps(additional_info, indent=2))
    return responses


data = []


def main(args):
    images = [item['image'] for item in data]
    responses_raw = infer(data, images)

    responses = []
    for (response, additional_info), item in zip(responses_raw, data):
        response = format_response(response, len(item['response']['reasoning']))
        response['additional_info'] = additional_info
        responses.append(response)

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

    with open(args.input) as f:
        data = [json.loads(line) for line in f]

    try:
        main(args)
    finally:
        if args.launch_locally:
            process.kill()
