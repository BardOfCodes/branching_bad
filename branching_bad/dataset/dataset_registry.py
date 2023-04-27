class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):

        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def get_dataset(cls, config, **kwargs):
        dataset_class = cls._registry[config.NAME]
        dataset = dataset_class(config, **kwargs)
        return dataset
