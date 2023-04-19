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
        return model