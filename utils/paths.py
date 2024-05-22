import os
from pathlib import Path
import sys
import pyrootutils

class Paths:
    def __init__(self):
        current_path = Path(__file__).resolve()
        self.root_dir = pyrootutils.setup_root(current_path.parent.parent, indicator=".project-root", pythonpath=True)
        os.chdir(self.root_dir)
        if str(self.root_dir) not in sys.path:
            sys.path.append(str(self.root_dir))

        self.Datasets = self.root_dir / 'Datasets'
        self.Configs = self.root_dir / 'Configs'
        self.Logs = self.root_dir / 'Logs'
        self.Results = self.root_dir / 'Results'
        self.make_sure()

    def make_sure(self):
        self.Datasets.mkdir(parents=True, exist_ok=True)
        self.Configs.mkdir(parents=True, exist_ok=True)
        self.Logs.mkdir(parents=True, exist_ok=True)
        self.Results.mkdir(parents=True, exist_ok=True)

    def chk_valid(self, data_path):
        result = True
        result &= (data_path / 'train').exists()
        result &= (data_path / 'test').exists()
        result &= (data_path / 'valid').exists()
        return result

    @staticmethod
    def path(name):
        return Path(name).resolve()

p = Paths()
