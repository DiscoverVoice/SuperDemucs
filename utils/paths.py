from pathlib import Path


class Paths:
    def __init__(self):
        current_path = Path(__file__)
        self.root_dir = current_path.parent.parent.resolve()
        self._Data = self.root_dir / 'Datasets'
        self._Config = self.root_dir / 'Configs'
        self.make_sure()

    def make_sure(self):
        Path.mkdir(self._Data, parents=True, exist_ok=True)
        Path.mkdir(self._Config, parents=True, exist_ok=True)

    @staticmethod
    def path(name):
        return Path(name)