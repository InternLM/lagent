import random

from termcolor import COLORS, colored

from lagent.utils import get_logger
from .hook import Hook


class MessageLogger(Hook):
    def __init__(self, name: str = 'lagent', add_file_handler: bool = False):
        self.logger = get_logger(
            name, 'info', '%(asctime)s %(levelname)8s %(name)8s - %(message)s', add_file_handler=add_file_handler
        )
        self.sender2color = {}

    def before_agent(self, agent, messages):
        for message in messages:
            self._process_message(message)

    def after_agent(self, agent, message):
        self._process_message(message)

    def before_action(self, executor, message):
        self._process_message(message)

    def after_action(self, executor, message):
        self._process_message(message)

    def _process_message(self, message):
        sender = message.sender
        color = self.sender2color.setdefault(sender, random.choice(list(COLORS)))
        msg_str = f'message sender: {sender}'
        if getattr(message, 'reasoning_content', None):
            msg_str += f'\nReasoning:{message.reasoning_content}'
        thinking = getattr(message, 'thinking', None)
        if thinking:
            msg_str += f'\nThinking:{thinking}'
        if getattr(message, 'content', None):
            msg_str += f'\nContent:{message.content}'
        if getattr(message, 'tool_calls', None):
            msg_str += f'\nTool Calls:{message.tool_calls}'
        raw_content = getattr(message, 'raw_content', None)
        if raw_content and raw_content != getattr(message, 'content', None):
            msg_str += f'\nRaw:{raw_content}'
        if getattr(message, 'reward', None) is not None:
            msg_str += f'\nReward:{message.reward}'
        finish_reason = getattr(message, 'finish_reason', None)
        if finish_reason:
            msg_str += f'\nFinishReason:{finish_reason}'
        stream_state = getattr(message, 'stream_state', None)
        if stream_state is not None:
            msg_str += f'\nStreamState:{stream_state}'
        extra_info = getattr(message, 'extra_info', None)
        if extra_info:
            msg_str += f'\nExtraInfo:{extra_info}'

        self.logger.info(colored(msg_str, color))
