# Copyright (c) OpenMMLab. All rights reserved.
import os

from lagent.llms.lmdepoly_wrapper import (LMDeployClient, LMDeployPipeline,
                                          LMDeployServer, TritonClient)


def run(mode='TritonClient', generation=True, stream=False):
    """An example to communicate with inference server through the command line
    interface."""

    prompt = """
[UNUSED_TOKEN_146]system
You are InternLM, a large language model trained by PJLab. Answer as concisely as possible. 当开启工具以及代码时，根据需求选择合适的工具进行调用[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]system name=[UNUSED_TOKEN_142]
你现在可以通过如下格式向 Jupyter Notebook 发送并执行代码:
[UNUSED_TOKEN_144][UNUSED_TOKEN_142]```python

代码

```

当遇到以下问题时，请使用上述格式调用 Jupyter Notebook 去解决，并根据执行结果做出友好的回复：
1. 文件操作和数据导入，比如处理CSV、JSON等格式文件
2. 数据分析或处理，比如数据操作或图像绘制如折线图、柱状图等
3. 数学相关的问题。当遇到数学问题时，你需要分析题目，并给出代码去解决这个题目[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]user
请帮我分析一下这份数据。[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]user name=file
{"path": "/temp/53247907C639A49C78D04A0255D4894C.csv", "size": "0.01 MB"}[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
"""

    if mode == 'TritonClient':
        chatbot = TritonClient(
            tritonserver_addr='10.140.0.65:33433',
            model_name='internlm2-chat-7b',
            session_len=16384 * 2,
            top_p=0.8,
            top_k=None,
            temperature=0,
            repetition_penalty=1.0,
            stop_words=['[UNUSED_TOKEN_145]', '[UNUSED_TOKEN_143]'])
        if generation:
            response = chatbot.generate(
                inputs=prompt,
                session_id=2967,
                request_id='1',
                max_out_len=1024,
                sequence_start=True,
                sequence_end=True,
            )
            print(response, end='\n', flush=True)
        else:
            if stream:
                for status, res, _ in chatbot.stream_chat(
                    inputs=prompt,
                    session_id=2967,
                    request_id='1',
                    max_out_len=1024,
                    sequence_start=True,
                    sequence_end=True,
                ):
                    print(res, end='\n', flush=True)
    elif mode == 'LMDeployClient':
        chatbot = LMDeployClient(
            path='internlm2-chat-7b',
            url='http://10.140.1.82:23333',
            top_p=0.8,
            top_k=100,
            temperature=0,
            repetition_penalty=1.0,
            stop_words=["[UNUSED_TOKEN_145]", "[UNUSED_TOKEN_143]"]
        )
        if generation:
            response = chatbot.generate(
                inputs=prompt,
                session_id=2967,
                sequence_start=True,
                sequence_end=True,
                ignore_eos=False,
                timeout=30,
                max_tokens=1024,
                top_k=100
            )
            print(response, end='\n', flush=True)
        else:
            if stream:
                for status, res, _ in chatbot.stream_chat(
                    inputs=prompt,
                    session_id=2967,
                    sequence_start=True,
                    sequence_end=True,
                    stream=True,
                    ignore_eos=False,
                    timeout=30,
                    max_tokens=1024,
                    top_k=100
                ):
                    print(res, end='\n', flush=True)
    elif mode == 'LMDeployPipeline':
        chatbot = LMDeployPipeline(
            path='internlm/internlm2-chat-7b',
            model_name='internlm2-chat-7b',
            top_k=100
        )
        prompt = prompt.replace('[UNUSED_TOKEN_146]', '<|im_start|>')
        prompt = prompt.replace('[UNUSED_TOKEN_142]', '<|interpreter|>')
        prompt = prompt.replace('[UNUSED_TOKEN_141]', '<|plugin|>')
        prompt = prompt.replace('[UNUSED_TOKEN_145]', '<|im_end|>')
        response = chatbot.generate(
            inputs=prompt
        )
        print(response, end='\n', flush=True)
    elif mode == 'LMDeployServer':
        chatbot = LMDeployServer(
            path='internlm/internlm2-chat-7b',
            model_name='internlm2-chat-7b',
            top_k=100,
        )
        prompt = prompt.replace('[UNUSED_TOKEN_146]', '<|im_start|>')
        prompt = prompt.replace('[UNUSED_TOKEN_142]', '<|interpreter|>')
        prompt = prompt.replace('[UNUSED_TOKEN_141]', '<|plugin|>')
        prompt = prompt.replace('[UNUSED_TOKEN_145]', '<|im_end|>')
        response = chatbot.generate(
            inputs=prompt
        )
        # print(response, end='\n', flush=True)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    run(mode='LMDeployServer', generation=True, stream=False)
