"""Abstract strategy module"""
import abc
import typing


class AbstractFeatureGenerator(abc.ABC):
    @abc.abstractmethod
    def __init__(self, config: typing.Dict):
        pass

    @abc.abstractmethod
    def generate_features(self, paths_dictionary: typing.Dict):
        pass

    @abc.abstractmethod
    def load_features(self, paths_dictionary: typing.Dict):
        pass
