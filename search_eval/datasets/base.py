"""Module contains super class for all datasets
"""

from pathlib import Path

class Dataset:
    def import_from_disk(self, dir:Path):
        pass

    def load_from_cache(self, dir:Path):
        pass

    def save_to_cache(self, dir:Path):
        pass
    
    def get_queries(self):
        pass

    def get_judgements(self, quiery_id:int):
        pass
