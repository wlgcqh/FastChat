from abc import ABC, abstractmethod
from typing import List, Any
from omegaconf import OmegaConf


class Pipeline(ABC):
    def __init__(self, config_file: str) -> None:
        self.cfg = OmegaConf.load(config_file)

    @abstractmethod
    def run(self, conv: Any):
        """
        conv: e.g., [{'user': xxx, 'assistant': xxx}]
        return:
            refined_conv: [{'user': xxx, 'assistant': xxx}]
            conv_info : {'turn_cnt': xxx, 'dedup_cnt': xxx ...}
        """
        conv_info = {}
        return conv, conv_info
