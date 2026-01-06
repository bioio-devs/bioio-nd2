import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import nd2
import numpy as np

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass(frozen=True)
class WellPosition:
    """
    Logical well identifier.
    """

    row: str
    col: str


@dataclass(frozen=True)
class PlateWell:
    """
    Physical well geometry.
    """

    row: str
    col: str
    center_x: float
    center_y: float


@dataclass(frozen=True)
class PlateSpec:
    """
    Physical and logical definition of a multiwell plate.
    """

    name: str
    rows: List[str]
    cols: List[str]
    plate_width_mm: float
    plate_height_mm: float
    a1_offset_mm: Tuple[float, float]
    well_spacing_um: float


###############################################################################
# Plate specifications
###############################################################################

PLATE_96 = PlateSpec(
    name="96",
    rows=list("ABCDEFGH")[::-1],
    cols=[str(i) for i in range(1, 13)],
    plate_width_mm=126.6,
    plate_height_mm=85.7,
    a1_offset_mm=(14.3, 11.36),
    well_spacing_um=9000.0,
)


def determine_plate_spec(
    rdr: nd2.ND2File,
) -> PlateSpec:
    """
    Determine plate specification from ND2 metadata.
    Currently only support 96 Well plate geometries.

    Policy
    ------
    - Error if no XYPosLoop is present (no way to know what well)
    - Assume 96-well plate unless there is evidence that
      the stage extents exceed what a 96-well could plausibly be.
    """
    # Tolerance for 96-well dimensions (µm)
    # Allows for partial scans and Stage positioning variation
    tolerance_um = 1000.0

    # Check XYPosLoop is present
    for exp in rdr.experiment:
        if "XYPosLoop" in str(exp):
            points = exp.parameters.points
            break
    else:
        raise RuntimeError(
            "Unable to determine plate geometry: "
            "ND2 file does not contain XYPosLoop metadata."
        )

    xs = np.array([-p.stagePositionUm.x for p in points])
    ys = np.array([-p.stagePositionUm.y for p in points])

    x_extent = xs.max() - xs.min()
    y_extent = ys.max() - ys.min()

    # Compute expected full 96-well plate extents
    full_96_x = (len(PLATE_96.cols) - 1) * PLATE_96.well_spacing_um
    full_96_y = (len(PLATE_96.rows) - 1) * PLATE_96.well_spacing_um

    # Accept if observed extents are not absurdly big for 96-well
    if x_extent <= full_96_x + tolerance_um and y_extent <= full_96_y + tolerance_um:
        return PLATE_96

    # If we get here we probably dont support the geometry
    raise RuntimeError(
        "ND2 stage extents exceed 96-well geometry. "
        f"Observed extent≈{x_extent:.0f}×{y_extent:.0f} µm "
        f"vs expected 96-well max≈{full_96_x:.0f}×{full_96_y:.0f} µm."
    )


def generate_plate_geometry(spec: PlateSpec) -> List[PlateWell]:
    """
    Generate plate geometry centered at (0, 0) from a PlateSpec.
    """
    plate_center = np.array([spec.plate_width_mm / 2, spec.plate_height_mm / 2]) * 1000
    a1_center = np.array(spec.a1_offset_mm) * 1000 - plate_center
    wells: List[PlateWell] = []

    for row_index, row in enumerate(spec.rows):
        for column_index, column in enumerate(spec.cols):
            center = a1_center + np.array(
                [column_index * spec.well_spacing_um, row_index * spec.well_spacing_um]
            )
            wells.append(
                PlateWell(
                    row=row,
                    col=column,
                    center_x=center[0],
                    center_y=center[1],
                )
            )

    return wells


def get_plate_geometry_from_nd2(
    rdr: nd2.ND2File,
) -> List[PlateWell]:
    """
    Determine and generate plate geometry from ND2 metadata.
    (Currently only 96-Well)
    """
    spec = determine_plate_spec(rdr)
    return generate_plate_geometry(spec)


###############################################################################
# Position Extraction
###############################################################################


def extract_position_stage_xy_um(
    rdr: nd2.ND2File,
) -> Dict[int, Tuple[float, float]]:
    """
    Extract stage XY (µm) for each ND2 position index.

    Returns
    -------
    Dict[int, Tuple[float, float]]
        Mapping of position_index -> (x_um, y_um)
    """
    for exp in rdr.experiment:
        if "XYPosLoop" in str(exp):
            points = exp.parameters.points
            break
    else:
        raise RuntimeError(
            "Unable to determine plate geometry: "
            "ND2 file does not contain XY position metadata."
        )

    position_xy: Dict[int, Tuple[float, float]] = {}

    for i, p in enumerate(points):
        position_xy[i] = (
            -p.stagePositionUm.x,
            -p.stagePositionUm.y,
        )

    return position_xy


def extract_scene_to_position_index(
    rdr: nd2.ND2File,
    num_scenes: int,
) -> Dict[int, int]:
    """
    Returns
    -------
    Dict[int, int]
        A dictionary mapping each scene index to its corresponding
        ND2 position index.
    """
    mapping: Dict[int, int] = {}

    for scene_index in range(num_scenes):
        fm = rdr.frame_metadata(scene_index)

        pos_index = (
            getattr(getattr(fm, "position", None), "index", None)
            or getattr(
                getattr(
                    getattr(fm, "channels", [None])[0],
                    "position",
                    None,
                ),
                "index",
                None,
            )
            or scene_index
        )

        mapping[scene_index] = pos_index

    return mapping


def find_closest_well(
    x: float,
    y: float,
    wells: Iterable[PlateWell],
) -> WellPosition:
    """
    Find the nearest well center to a stage (x, y) position.
    """
    best = min(
        wells,
        key=lambda w: (x - w.center_x) ** 2 + (y - w.center_y) ** 2,
    )
    return WellPosition(row=best.row, col=best.col)


def map_scenes_to_wells(
    scene_to_position: Dict[int, int],
    position_xy: Dict[int, Tuple[float, float]],
    wells: Iterable[PlateWell],
) -> Dict[int, WellPosition]:
    """
    Map absolute scene indices to logical well positions.
    """
    mapping: Dict[int, WellPosition] = {}

    for scene_index, pos_index in scene_to_position.items():
        x, y = position_xy[pos_index]
        mapping[scene_index] = find_closest_well(x, y, wells)

    return mapping
