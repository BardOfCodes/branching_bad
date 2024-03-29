import _pickle as cPickle
class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):

        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create_model(cls, config):
        model_class = cls._registry[config.NAME]
        model = model_class(config)
        if config.LOAD_WEIGHTS is not None:
            save_info = cPickle.load(open(config.LOAD_WEIGHTS, "rb"))
            weights = save_info["model"]
            model.load_state_dict(weights)
        return model
