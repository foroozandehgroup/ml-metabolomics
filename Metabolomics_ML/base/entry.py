from dataclasses import dataclass
import numpy as np

@dataclass
class Entry:
    id: int
    class_: str
    integs: np.ndarray