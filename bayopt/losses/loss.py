import abc
class Loss:
    @abc.abstractmethod
    def evaluate(self, k):
        pass

    def reset(self):
        pass
