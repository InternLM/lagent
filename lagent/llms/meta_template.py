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

INTERNLM_META = [
    dict(role='system', begin='<|System|>:', end='\n'),
    dict(role='user', begin='<|User|>:', end='\n'),
    dict(role='assistant', begin='<|Bot|>:', end='<eoa>\n'),
]
