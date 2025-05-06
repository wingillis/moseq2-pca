import cv2
import tempfile
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
from toolz import dissoc, keyfilter


@dataclass
class MouseProcessingParams:
    """Parameters for general data processing"""

    min_height: int = 10
    max_height: int = 120
    gaussfilter_space: tuple[float, float] = (1.5, 1.0)
    tailfilter_size: tuple[int, int] = (9, 9)
    tailfilter_shape: Literal["ellipse", "rectangle"] = "ellipse"
    tail_threshold: int = 5
    medfilter_space: Optional[tuple[int, int]] = None
    medfilter_time: Optional[tuple[int, int]] = None
    gaussfilter_time: float = 0.0

    tailfilter: np.ndarray = field(init=False)

    def __post_init__(self):
        shape_map = {"ellipse": cv2.MORPH_ELLIPSE, "rectangle": cv2.MORPH_RECT}
        self.tailfilter = cv2.getStructuringElement(
            shape_map[self.tailfilter_shape], self.tailfilter_size
        )


@dataclass
class ChangepointParams:
    """Parameters for changepoint detection"""

    klags: int = 6
    sigma: float = 3.5
    threshold: float = 0.5
    neighbors: int = 1


@dataclass
class MaskParams:
    """Parameters for mask-based missing data handling"""

    mask_height_threshold: float = 5.0
    mask_threshold: float = -16.0


@dataclass
class DaskConfig:
    """Configuration for Dask cluster"""

    nworkers: int = 50
    processes: int = 1
    memory: str = "4GB"
    cores: int = 1
    wall_time: str = "01:00:00"
    queue: str = "debug"
    local_processes: bool = False
    cluster_type: Literal["local", "slurm"] = "local"
    timeout: int = 10
    cache_path: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "moseq2_pca")
    dashboard_port: str = "8787"


@dataclass
class SVDConfig:
    """Configuration for SVD"""

    # rank (int): Rank of the desired thin SVD decomposition.
    rank: int = 10
    # iters (int): Number of SVD iterations
    iters: int = 10
    # recon_pcs (int): Number of PCs to reconstruct for missing data.
    recon_pcs: int = 10
    # missing_data (bool): Whether to use missing data for SVD.
    missing_data: bool = False

    chunk_size: int = 4000

    # set to True if you want to use memory efficient SVD when data is larger than available memory
    # generally keep False
    memory_efficient: bool = False

    def __post_init__(self):
        if self.missing_data:
            print('Detected missing data')


def create_dataclass_from_dict(dataclass_type, data_dict):
    """
    Create a dataclass from a dictionary. Remove any keys that are not in the dataclass.
    Return the dataclass and filtered dictionary.
    """
    # get the fields of the dataclass
    fields = dataclass_type.__dataclass_fields__
    # filter the dictionary to only include the fields of the dataclass
    dataclass_out = dataclass_type(**keyfilter(lambda k: k in fields, data_dict))
    # filter the dictionary to only include the fields of the dataclass
    filtered_dict = dissoc(data_dict, *fields)

    return dataclass_out, filtered_dict
