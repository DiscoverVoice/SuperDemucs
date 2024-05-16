# In[]
import os
import pathlib
from pathlib import Path

current_path = Path(__file__)


class Paths:
    def __init__(self):
        self.root_dir = current_path.parent.parent.resolve()
        self.data_dir = Path(self.root_dir / 'Datasets')
        self.make_sure()

    def make_sure(self):
        Path.mkdir(self.data_dir, parents=True, exist_ok=True)

p = Paths()
