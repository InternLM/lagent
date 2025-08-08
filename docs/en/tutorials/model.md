# Model

## Class TritonClient

Chatbot for LLaMA series models with turbomind as inference engine.

Here's an example:

```python
from lagent.llms.lmdepoly_wrapper import TritonClient
from lagent.llms.meta_template import INTERNLM2_META as META
chatbot = TritonClient(
    tritonserver_addr='Your service addr',
    model_name='internlm2-chat-7b',
    meta_template=META,
    session_len=16384 * 2,
    max_new_tokens=1024,
    top_p=0.8,
    top_k=None,
    temperature=0,
    repetition_penalty=1.0,
    stop_words=['<|im_end|>']
)
```

- `tritonserver_addr` (str): the address in format `ip:port` of triton inference server
- `model_name` (str): needed when model_path is a pytorch model on huggingface.co, such as `internlm2-chat-7b`, `Qwen-7B-Chat`, `Baichuan2-7B-Chat` and so on.
- `meta_template` (List\[dict\]): Prefix and suffix for specific role. Here we resort to it to construct the input prompt.
- `session_len` (int): The max context size of model.
- `max_new_tokens` (int):  The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
- `top_p` (float): If set to float \< 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
- `top_k` (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
- `temperature` (float): Controls randomness, higher values increase diversity.
- `repetition_penalty` (float): The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
- `stop_words` (Union\[List\[str\], str\]): Symbols, when encountered by the model stop sequence generation.

### generate

Generates sequences of token ids for models with a language modeling head.

<Tip warning={true}>

Most generation-controlling parameters, if not passed, will be set to the
base model's default generation params. You can override any of them `max_new_tokens, top_p, top_k, temperature, repetition_penalty, stop_words` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, max_new_tokens=2048, temperature=0.8)`.

</Tip>

Here's an example:

```python
response = chatbot.generate(
    inputs='hello',
    session_id=2967,
    request_id='1',
    sequence_start=True,
    sequence_end=True,
    max_new_tokens=2048,
    temperature=0.8
)
```

- `inputs` (Union\[str, List\[str\]\]): The sequence used as a prompt for the generation or as model inputs to the encoder. Here we also support batch input.
- `session_id` (int): the identical id of a session.
- `request_id` (str): the identical id of this round conversation.
- `sequence_start` (bool): indicator for starting a sequence.
- `sequence_end` (bool): indicator for ending a sequence.
- `max_new_tokens` (int): The maximum numbers of tokens to generate. Here we override the default one.
- `temperature` (float): Controls randomness, higher values increase diversity. Here we override the default one.

### stream_chat

Token streaming is the mode in which the server returns the tokens one by one as the model generates them. As the function name suggests, it supports messages as input, similar to openai.ChatCompletion, and streams a sequence of outputs.

<Tip warning={true}>

Most generation-controlling parameters, if not passed, will be set to the
base model's default generation params. You can override any of them `max_new_tokens, top_p, top_k, temperature, repetition_penalty, stop_words` by passing the corresponding
parameters to generate(), e.g. `.stream_chat(inputs, max_new_tokens=2048, temperature=0.8)`.

</Tip>

Here's an example:

```python
for status, res, _ in chatbot.stream_chat(
    inputs=[
        dict(
            role='system',
            content='You are InternLM (书生·浦语), a helpful, honest, and harmless AI assistant developed by Shanghai AI Laboratory (上海人工智能实验室).',
        ),
        dict(
            role='user',
            content='hello'
        )
    ],
    session_id=2967,
    request_id='1',
    sequence_start=True,
    sequence_end=True,
    max_new_tokens=2048,
    temperature=0.8
):
    print(res, end='\n', flush=True)
```

- `inputs` (List\[dict\]): The messages used as a prompt for the generation or as model inputs to the encoder.
- `session_id` (int): the identical id of a session.
- `request_id` (str): the identical id of this round conversation.
- `sequence_start` (bool): indicator for starting a sequence.
- `sequence_end` (bool): indicator for ending a sequence.
- `max_new_tokens` (int): The maximum numbers of tokens to generate. Here we override the default one.
- `temperature` (float): Controls randomness, higher values increase diversity. Here we override the default one.

## Class LMDeployPipeline

Here's an example:

```python
from lagent.llms.lmdepoly_wrapper import LMDeployPipeline
from lagent.llms.meta_template import INTERNLM2_META as META
chatbot = LMDeployPipeline(
    path='internlm/internlm2-chat-7b',
    model_name='internlm2-chat-7b',
    meta_template=META,
    max_new_tokens=1024,
    top_p=0.8,
    top_k=None,
    temperature=0,
    repetition_penalty=1.0,
    stop_words=['<|im_end|>']
)
```

- `path` (str): the path of a model.
  It could be one of the following options:
  - i) A local directory path of a turbomind model which is converted by `lmdeploy convert` command or download from ii) and iii).
  - ii) The model_id of a lmdeploy-quantized model hosted inside a model repo on huggingface.co, such as `InternLM/internlm-chat-20b-4bit`, `lmdeploy/llama2-chat-70b-4bit`, etc.
  - iii) The model_id of a model hosted inside a model repo on huggingface.co, such as `internlm/internlm2-chat-7b`, `Qwen/Qwen-7B-Chat`, `baichuan-inc/Baichuan2-7B-Chat` and so on.
- `model_name` (str): needed when model_path is a pytorch model on huggingface.co, such as `internlm2-chat-7b`, `Qwen-7B-Chat`, `Baichuan2-7B-Chat` and so on.
- `meta_template` (List\[dict\]): Prefix and suffix for specific role. Here we resort to it to construct the input prompt.
- `max_new_tokens` (int):  The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
- `top_p` (float): If set to float \< 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
- `top_k` (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
- `temperature` (float): Controls randomness, higher values increase diversity.
- `repetition_penalty` (float): The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
- `stop_words` (Union\[List\[str\], str\]): Symbols, when encountered by the model stop sequence generation.

### generate

Generates sequences of token ids for models with a language modeling head.

<Tip warning={true}>

Most generation-controlling parameters, if not passed, will be set to the
base model's default generation params. You can override any of them `max_new_tokens, top_p, top_k, temperature, repetition_penalty, stop_words` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, max_new_tokens=2048, temperature=0.8)`.

</Tip>

```python
response = chatbot.generate(
    inputs='hello',
    max_new_tokens=2048,
    temperature=0.8
)
```

- `inputs` (Union\[str, List\[str\]\]): The sequence used as a prompt for the generation or as model inputs to the encoder. Here we also support batch input.
- `max_new_tokens` (int): The maximum numbers of tokens to generate. Here we override the default one.
- `temperature` (float): Controls randomness, higher values increase diversity. Here we override the default one.

## Class LMDeployServer

This will run the api server in a subprocess.

Here's an example:

```python
from lagent.llms.lmdepoly_wrapper import LMDeployServer
from lagent.llms.meta_template import INTERNLM2_META as META
chatbot = LMDeployServer(
    path='internlm/internlm2-chat-7b',
    model_name='internlm2-chat-7b',
    server_name='0.0.0.0',
    server_port=23333,
    tp=1,
    meta_template=META,
    max_new_tokens=1024,
    top_p=0.8,
    top_k=None,
    temperature=0,
    repetition_penalty=1.0,
    stop_words=['<|im_end|>']
)
```

- `path` (str): the path of a model.
  It could be one of the following options:
  - i) A local directory path of a turbomind model which is converted by `lmdeploy convert` command or download from ii) and iii).
  - ii) The model_id of a lmdeploy-quantized model hosted inside a model repo on huggingface.co, such as `InternLM/internlm-chat-20b-4bit`, `lmdeploy/llama2-chat-70b-4bit`, etc.
  - iii) The model_id of a model hosted inside a model repo on huggingface.co, such as `internlm/internlm2-chat-7b`, `Qwen/Qwen-7B-Chat`, `baichuan-inc/Baichuan2-7B-Chat` and so on.
- `model_name` (str): needed when model_path is a pytorch model on huggingface.co, such as `internlm2-chat-7b`, `Qwen-7B-Chat`, `Baichuan2-7B-Chat` and so on.
- `server_name` (str): host ip for serving. Defaults to `0.0.0.0`
- `server_port` (int): server port. Defaults to `23333`
- `tp` (int): Num of gpus that tensor parallel on.
- `meta_template` (List\[dict\]): Prefix and suffix for specific role. Here we resort to it to construct the input prompt.
- `max_new_tokens` (int):  The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
- `top_p` (float): If set to float \< 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
- `top_k` (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
- `temperature` (float): Controls randomness, higher values increase diversity.
- `repetition_penalty` (float): The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
- `stop_words` (Union\[List\[str\], str\]): Symbols, when encountered by the model stop sequence generation.

### generate

```python
response = chatbot.generate(
    inputs='hello',
    session_id=2967,
    sequence_start=True,
    sequence_end=True,
    ignore_eos=False,
    timeout=30,
    max_new_tokens=2048,
    temperature=0.8
)
```

- `ignore_eos` (bool): Whether to ignore the EOS token and continue generating tokens after the EOS token is generated. Defaults to False.
- `timeout` (int): max time to wait for response. Defaults to 30s.

### stream_chat

```python
for status, res, _ in chatbot.stream_chat(
    inputs=[
        dict(
            role='system',
            content='You are InternLM (书生·浦语), a helpful, honest, and harmless AI assistant developed by Shanghai AI Laboratory (上海人工智能实验室).',
        ),
        dict(
            role='user',
            content='hello'
        )
    ],
    session_id=2967,
    sequence_start=True,
    sequence_end=True,
    stream=True,
    ignore_eos=False,
    timeout=30,
    max_new_tokens=2048,
    temperature=0.8
):
    print(res, end='\n', flush=True)
```

## Class LMDeployClient

Chatbot for LLaMA series models with turbomind as inference engine.

Here's an example:

```python
from lagent.llms.lmdepoly_wrapper import LMDeployClient
from lagent.llms.meta_template import INTERNLM2_META as META
chatbot = LMDeployClient(
    url='Your service addr',
    model_name='internlm2-chat-7b',
    meta_template=META,
    max_new_tokens=1024,
    top_p=0.8,
    top_k=None,
    temperature=0,
    repetition_penalty=1.0,
    stop_words=['<|im_end|>']
)
```

### generate

```python
response = chatbot.generate(
    inputs='hello',
    session_id=2967,
    sequence_start=True,
    sequence_end=True,
    ignore_eos=False,
    timeout=30,
    max_new_tokens=2048,
    temperature=0.8
)
```

### stream_chat

```python
for status, res, _ in chatbot.stream_chat(
    inputs=[
        dict(
            role='system',
            content='You are InternLM (书生·浦语), a helpful, honest, and harmless AI assistant developed by Shanghai AI Laboratory (上海人工智能实验室).',
        ),
        dict(
            role='user',
            content='hello'
        )
    ],
    session_id=2967,
    sequence_start=True,
    sequence_end=True,
    stream=True,
    ignore_eos=False,
    timeout=30,
    max_new_tokens=2048,
    temperature=0.8
):
    print(res, end='\n', flush=True)
```
