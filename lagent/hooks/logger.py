import random
from typing import Optional

from termcolor import COLORS, colored

from lagent.utils import get_logger
from .hook import Hook


class MessageLogger(Hook):

    def __init__(self, name: str = 'lagent'):
        self.logger = get_logger(name, 'info')
        self.sender2color = {}

    def before_agent(self, agent, messages, session_id):
        for message in messages:
            self._process_message(message, session_id)

    def after_agent(self, agent, message, session_id):
        self._process_message(message, session_id)

    def before_action(self, executor, message, session_id):
        self._process_message(message, session_id)

    def after_action(self, executor, message, session_id):
        self._process_message(message, session_id)

    def _process_message(self, message, session_id):
        sender = message.sender
        color = self.sender2color.setdefault(sender,
                                             random.choice(list(COLORS)))
        self.logger.info(
            colored(
                f'session id: {session_id}, message sender: {sender}\n'
                f'{message.content}', color))
