META = [
    dict(
        role='system',
        begin=dict(
            with_name='[UNUSED_TOKEN_146]system name={name}\n',
            without_name='[UNUSED_TOKEN_146]system\n',
            name={
                'interpreter': '[UNUSED_TOKEN_142]',
                'plugin': '[UNUSED_TOKEN_141]',
            }),
        end='[UNUSED_TOKEN_145]\n',
    ),
    dict(
        role='user',
        begin=dict(
            with_name='[UNUSED_TOKEN_146]user name={name}\n',
            without_name='[UNUSED_TOKEN_146]user\n',
        ),
        end='[UNUSED_TOKEN_145]\n'),
    dict(
        role='assistant',
        begin=dict(
            with_name='[UNUSED_TOKEN_146]assistant name={name}\n',
            without_name='[UNUSED_TOKEN_146]assistant\n',
            name={
                'interpreter': '[UNUSED_TOKEN_142]',
                'plugin': '[UNUSED_TOKEN_141]',
            }),
        end='[UNUSED_TOKEN_145]\n'),
    dict(
        role='environment',
        begin=dict(
            with_name='[UNUSED_TOKEN_146]environment name={name}\n',
            without_name='[UNUSED_TOKEN_146]environment\n',
            name={
                'interpreter': '[UNUSED_TOKEN_142]',
                'plugin': '[UNUSED_TOKEN_141]',
            }),
        end='[UNUSED_TOKEN_145]\n'),
]
