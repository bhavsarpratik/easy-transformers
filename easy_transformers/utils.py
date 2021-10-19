from transformers import AutoConfig


def from_pretrained(pretrained_model_name_or_path, **kwargs):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    patch_call(config, get_hash)
    return config


def patch_call(instance, func, memo={}):
    if type(instance) not in memo:

        class _(type(instance)):
            def __hash__(self, *arg, **kwargs):
                return func(self, *arg, **kwargs)

        memo[type(instance)] = _

    instance.__class__ = memo[type(instance)]


def get_hash(self):
    return hash(self.name_or_path)
