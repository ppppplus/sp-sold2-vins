from abc import ABC, abstractmethod

class BaseExtractModel(ABC):
    def __init__(self, params):
        self._init(params)
    @abstractmethod
    def extract(self, img):
      return NotImplementedError
    
class BaseMatchModel(ABC):
    def __init__(self, params):
        self._init(params)
    @abstractmethod
    def match(self, data):
      return NotImplementedError