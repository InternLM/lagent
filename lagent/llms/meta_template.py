INTERNLM2_META = [
    dict(
        role='system',
        begin=dict(
            with_name='<|im_start|>system name={name}\n',
            without_name='<|im_start|>system\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            }),
        end='<|im_end|>\n',
    ),
    dict(
        role='user',
        begin=dict(
            with_name='<|im_start|>user name={name}\n',
            without_name='<|im_start|>user\n',
        ),
        end='<|im_end|>\n'),
    dict(
        role='assistant',
        begin=dict(
            with_name='<|im_start|>assistant name={name}\n',
            without_name='<|im_start|>assistant\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            }),
        end='<|im_end|>\n'),
    dict(
        role='environment',
        begin=dict(
            with_name='<|im_start|>environment name={name}\n',
            without_name='<|im_start|>environment\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            }),
        end='<|im_end|>\n'),
]

LLAMA2_AGENT_META = [
    dict(role='system', begin='<|Human|>െ', end='\n '),
    dict(role='user', begin='<|Human|>െ', end='\n '),
    dict(role='assistant', begin='<|Assistant|>െ', end='ി\n '),
]

LLAMA3_META = [
    dict(
        role='system',
        begin='<|start_header_id|>system<|end_header_id|>\n\n',
        end='<|eot_id|>'),
    dict(
        role='user',
        begin='<|start_header_id|>user<|end_header_id|>\n\n',
        end='<|eot_id|>'),
    dict(
        role='assistant',
        begin='<|start_header_id|>assistant<|end_header_id|>\n\n',
        end='<|eot_id|>'),
]
