"""
Node template for creating custom nodes.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

import math
import numpy as np


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")
        self.fade_in = "start"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        if self.fade_in == "start":
            img = inputs["img"]
            # use input source dimensions
            self.max_width = img.shape[1]
            self.max_height = img.shape[0]
            self.fade_in = "anim"
            self.size_pct = 1  # 100 == 100% == fade in done
        elif self.fade_in == "anim":
            if self.size_pct < 100:
                self.size_pct += 1
            else:
                self.fade_in = "done"

        # render output based on size_pct
        width = math.floor(self.size_pct * self.max_width / 100)
        height = math.floor(self.size_pct * self.max_height / 100)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        outputs = {"img": img}
        return outputs
