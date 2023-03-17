from abc import ABC


class BaseSearch(ABC):
    def run_search(self, launcher, **kwargs):
        raise NotImplementedError()