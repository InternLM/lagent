class BaseAction:

    def __init__(self,
                 description=None,
                 name=None,
                 enable=True,
                 disable_description=None):
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self._description = description
        self._disable_description = disable_description
        self._enable = enable

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.name}:{self.description}'

    def __str__(self):
        return self.__repr__()

    def run(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    @property
    def enable(self):
        return self._enable

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        if self.enable:
            return self._description
        else:
            return self._disable_description
