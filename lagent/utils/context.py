import contextvars


class TokenWrapper:
    def __init__(self, var, token):
        self._var = var
        self._token = token

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._var.reset(self._token)

    @property
    def var(self):
        return self._token.var

    @property
    def old_value(self):
        return self._token.old_value


class ContextVar:
    """A wrapper around contextvars.ContextVar to support 'with var.set(val):'"""

    def __init__(self, name, default=contextvars.Token.MISSING):
        if default is contextvars.Token.MISSING:
            self._var = contextvars.ContextVar(name)
        else:
            self._var = contextvars.ContextVar(name, default=default)

    @property
    def name(self):
        return self._var.name

    def get(self, default=contextvars.Token.MISSING):
        if default is contextvars.Token.MISSING:
            return self._var.get()
        return self._var.get(default)

    def set(self, value):
        token = self._var.set(value)
        return TokenWrapper(self._var, token)

    def reset(self, token):
        if isinstance(token, TokenWrapper):
            token = token._token
        self._var.reset(token)
