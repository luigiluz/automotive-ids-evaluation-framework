"""Abstract model test module"""
import abc
import typing


class AbstractModelTest(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model, model_config_dict: typing.Dict):
        pass

    @abc.abstractmethod
    def execute(self, data):
        pass
