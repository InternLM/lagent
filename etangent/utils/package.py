import importlib


def is_module_exist(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False
    else:
        return True
