from abc import ABC


class BaseLauncher(ABC):
    def run_trial(self, **kwargs):
        raise NotImplementedError()