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
        if getattr(message, 'content', None):
            msg_str += f'\nContent:{message.content}'
        if getattr(message, 'tool_calls', None):
            msg_str += f'\nTool Calls:{message.tool_calls}'
            
        self.logger.info(colored(msg_str, color))
