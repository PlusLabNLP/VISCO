import base64
import multiprocessing
import socket
import subprocess
import time
from io import BytesIO

import anthropic
import google.generativeai as genai
import torch
import tqdm
from PIL import Image
from openai import OpenAI

args = None


def gemini_infer(image, query):
    try:
        model = genai.GenerativeModel(model_name=args.model)
        image_ = Image.open(BytesIO(base64.b64decode(image)))
        response = model.generate_content([query, image_])
        return response.text
    except:
        time.sleep(1)
        try:
            model = genai.GenerativeModel(model_name=args.model)
            image_ = Image.open(BytesIO(base64.b64decode(image)))
            response = model.generate_content([query, image_])
            return response.text
        except:
            print("Warning! gemini infer does not work")
            return "TODO"


def claude_func(image, query):
    client = anthropic.Anthropic(api_key=args.api_key)

    image_data = base64.b64decode(image)
    with BytesIO(image_data) as img_buffer:
        img = Image.open(img_buffer).convert("RGB")
        with BytesIO() as output_buffer:
            img.save(output_buffer, format='JPEG')
            image_str = base64.b64encode(output_buffer.getvalue()).decode('utf8')

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_str,
                    },
                },
                {
                    "type": "text",
                    "text": query,
                }
            ],
        }
    ]

    try:
        completion = client.messages.create(
            model=args.model,
            max_tokens=512,
            messages=messages,
        )
    except Exception as e:
        print("Error")
        print(e)
        time.sleep(60)
        completion = client.messages.create(
            model=args.model,
            max_tokens=512,
            messages=messages,
        )

    return completion.content[0].text


def func(obj):
    if len(obj) == 2:
        image, query = obj
        response2 = query2 = None
    else:
        assert len(obj) == 4
        image, query, response2, query2 = obj

    if args.model.startswith("gemini"):
        return gemini_infer(image, query)
    elif args.model.startswith("claude"):
        return claude_func(image, query)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
            ],
        },
    ]
    if image is not None:
        messages[0]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}",
            },
        })

    if response2 is not None:
        assert query2 is not None
        messages += [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": response2},
            ],
        }, {
            "role": "user",
            "content": [
                {"type": "text", "text": query2},
            ],
        }]
    else:
        assert query2 is None

    if args.model == 'auto':
        model = client.models.list().data[0].id
    else:
        model = args.model

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
    except:
        time.sleep(1)
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
        except Exception as e:
            print("Warning! infer does not work")
            print("Error:")
            print(e)
            return "TODO"

    return completion.choices[0].message.content


def get_pool(n_proc):
    class DummyPool:
        imap = map

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            pass

    if n_proc == 0:
        return DummyPool()
    else:
        return multiprocessing.Pool(n_proc)


def infer(queries, images, given_args):
    global args
    args = given_args

    if args.model.startswith("gemini"):
        genai.configure(api_key=args.api_key)

    responses = []
    assert len(images) == len(queries)
    with get_pool(args.n_proc) as p:
        for response in tqdm.tqdm(p.imap(func, zip(images, queries)), total=len(images)):
            responses.append(response)
            if len(responses) <= 5:
                print("\n--- Example output:", len(responses))
                print(responses[-1])

    return responses


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        return s.getsockname()[1]


def launch_locally(backend, model):
    port = find_available_port()

    if backend == 'lmdeploy':
        cmd = ['lmdeploy', 'serve', 'api_server', model, '--server-port', str(port),
               '--tp', str(torch.cuda.device_count()), ]
        if 'prometheus' in model:
            cmd += ['--chat-template', 'llava-v1', ]
    elif backend == 'vllm':
        cmd = ['vllm', 'serve', model, '--port', str(port), '--dtype', 'auto', '--api-key', 'YOUR_API_KEY',
               '--trust-remote-code', '--tensor-parallel-size', str(torch.cuda.device_count()), ]
        if '3.2' in model or 'nvlm' in model.lower():
            cmd += ['--enforce-eager', '--max-num-seqs', '32', '--max_model_len', '40000', ]
    else:
        assert backend == 'sglang'
        cmd = ['python', '-m', 'sglang.launch_server', '--model-path', model, '--port', str(port),
               '--chat-template=chatml-llava', ]
        if '7b' in model.lower():
            tp = 1
            if 'llava-critic' in model:
                cmd += ['--tokenizer-path', 'lmms-lab/llava-onevision-qwen2-7b-ov', ]
        elif '11b' in model.lower() or '13b' in model.lower():
            tp = 2
        else:
            assert '72b' in model.lower()
            tp = 4
            if 'llava-critic' in model:
                cmd += ['--tokenizer-path', 'lmms-lab/llava-onevision-qwen2-72b-ov-sft', ]
        assert torch.cuda.device_count() % tp == 0
        dp = torch.cuda.device_count() // tp
        cmd += ['--tp', str(tp), '--dp', str(dp), ]

    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)

    while True:
        try:
            _ = OpenAI(api_key='YOUR_API_KEY', base_url=f'http://0.0.0.0:{port}/v1').models.list()
            print("> launched. Proceed")
            return process, port
        except:
            pass
        time.sleep(5)


def get_answer_format(item):
    DEFAULT_ANSWER = "a single number, word or phrase"
    BINARY_ANSWER = "either \"Yes\" or \"No\""
    MULTI_CHOICE_ANSWER = 'in letter form of the choice selected, e.g., "A", "B", "C", "D"'
    INTEGER_ANSWER = "an integer number, e.g. 1, 2, 3"
    dataset = item['meta_data']['src_dataset']

    if dataset in ["VSR", "FigureQA", "POPE", "HallusionBench", ]:
        # a few datasets with only yes or no
        return BINARY_ANSWER

    elif dataset in ["WeMath", "ScienceQA", "SceMQA", "EmbSpatial", ]:
        # a few datasets completely multi-choice
        return MULTI_CHOICE_ANSWER

    elif dataset == "PlotQA":
        if item['label'] in ['Yes', 'No', ]:
            return BINARY_ANSWER
        else:
            return DEFAULT_ANSWER

    elif dataset == "MathVista":
        if item['meta_data']["question_type"] == "multi_choice":
            return MULTI_CHOICE_ANSWER
        elif item['meta_data']["answer_type"] == "integer":
            return INTEGER_ANSWER
        elif item['meta_data']["answer_type"] == "float":
            return "a decimal number, e.g., 1.23, 1.34, 1.45"
        else:
            assert item['meta_data']["answer_type"] == "list"
            return "a list, e.g., [1, 2, 3], [1.2, 1.3, 1.4]"

    elif dataset == "MMVet":
        return DEFAULT_ANSWER

    elif dataset == "MathVision":
        return "an integer number, e.g. -5, a decimal number, e.g. 3.5, or a coordinate, e.g. (1, 2)"

    elif dataset == "MMMU":
        if item["meta_data"]["question_type"] == "multiple-choice":
            return MULTI_CHOICE_ANSWER
        else:
            return DEFAULT_ANSWER

    elif dataset == "TallyQA":
        return INTEGER_ANSWER

    elif dataset in ["TextVQA", "DocVQA", ]:
        return "a word, a phrase, or a short concise sentence"

    elif dataset == "GQA":
        return DEFAULT_ANSWER

    elif dataset == "CLEVR":
        try:
            _ = int(item['label'])
            return INTEGER_ANSWER
        except:
            if item['label'] in ['yes', 'no', ]:
                return BINARY_ANSWER
            else:
                return DEFAULT_ANSWER

    elif dataset == "ChartQA":
        return DEFAULT_ANSWER

    else:
        raise ValueError(f"Dataset not found: {dataset}")
