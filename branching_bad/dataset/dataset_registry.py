class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):

        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def get_dataset(cls, config, subset, device):
        dataset_class = cls._registry[config.NAME]
        dataset = dataset_class(config, subset=subset, device=device)
        return dataset
