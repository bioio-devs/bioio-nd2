import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import nd2
import numpy as np

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass(frozen=True)
class WellPosition:
    """
    Logical well identifier.

    This represents a *logical* plate position (e.g. "A1", "H12") and does
    not encode any physical geometry. It is intentionally lightweight so it
    can be stored directly in standardized metadata.
    """

    row: str
    col: str


@dataclass(frozen=True)
class PlateWell:
    """
    Physical well geometry.

    Represents a single well on a plate with a physical center location in
    stage coordinates (µm). This object is used internally for nearest-neighbor
    calculations during well assignment.
    """

    row: str
    col: str
    center_x: float
    center_y: float


###############################################################################
# Assignment policy
###############################################################################


class WellAssignmentMode(str, Enum):
    """
    Policy for assigning stage positions to wells.

    These modes control how strictly a stage position must correspond to a
    physical well center in order to be assigned.
    """

    CLOSEST = "closest"
    """
    Always assign the nearest well, regardless of distance.

    This is the most permissive mode and matches historical behavior. It is
    appropriate when stage positions are known to be well-behaved or when
    approximate mapping is sufficient.
    """

    WITHIN_WELL = "within_well"
    """
    Assign only if the position falls within the physical well radius.

    This mode enforces a stricter physical interpretation of the plate geometry
    and will return `None` for positions that fall outside the defined well.
    """

    HALF_SPACING = "half_spacing"
    """
    Assign only if the position is within half the inter-well spacing.

    This is a compromise between strict physical containment and permissive
    nearest-neighbor assignment.
    """


###############################################################################
# Plate Definition
###############################################################################


class Plate:
    """
    Physical and logical definition of a multi-well plate.

    A `Plate` object encapsulates all information required to map ND2 XY stage
    positions to logical plate wells. Both geometry and assignment behavior are
    explicitly defined and may be configured by the user.

    Parameters
    ----------
    name : str
        Human-readable identifier for the plate geometry (e.g. "96", "384").

    rows : List[str]
        Ordered list of row identifiers (e.g. ["A", "B", ..., "H"]).
        The order defines the physical row layout on the plate.

    cols : List[str]
        Ordered list of column identifiers (e.g. ["1", "2", ..., "12"]).
        The order defines the physical column layout on the plate.

    plate_width_mm : float
        Physical width of the plate in millimeters (X dimension).

    plate_height_mm : float
        Physical height of the plate in millimeters (Y dimension).

    a1_offset_mm : Tuple[float, float]
        Offset (X, Y) in millimeters from the top-left corner of the plate
        to the center of well A1.

    well_spacing_um : float
        Center-to-center distance between adjacent wells, in microns (µm).

    well_radius_um : float
        Physical radius of a single well, in microns (µm).

    assignment_mode : WellAssignmentMode, optional
        Policy controlling how stage positions are assigned to wells.

        Default: WellAssignmentMode.CLOSEST
    """

    def __init__(
        self,
        *,
        name: str,
        rows: List[str],
        cols: List[str],
        plate_width_mm: float,
        plate_height_mm: float,
        a1_offset_mm: Tuple[float, float],
        well_spacing_um: float,
        well_radius_um: float,
        assignment_mode: WellAssignmentMode = WellAssignmentMode.CLOSEST,
    ) -> None:
        # Logical layout
        self.name = name
        self.rows = rows
        self.cols = cols

        # Physical geometry
        self.plate_width_mm = plate_width_mm
        self.plate_height_mm = plate_height_mm
        self.a1_offset_mm = a1_offset_mm
        self.well_spacing_um = well_spacing_um
        self.well_radius_um = well_radius_um

        # Assignment policy
        self.assignment_mode = assignment_mode

    @property
    def expected_extent_um(self) -> Tuple[float, float]:
        """
        Expected full plate extent in microns (X, Y).

        Returns
        -------
        Tuple[float, float]
            The expected plate extent in microns as (x_extent_um, y_extent_um).
        """
        return (
            (len(self.cols) - 1) * self.well_spacing_um,
            (len(self.rows) - 1) * self.well_spacing_um,
        )

    @property
    def half_spacing_um(self) -> float:
        """
        Half the distance between neighboring well centers.

        Used by the HALF_SPACING assignment mode.

        Returns
        -------
        float
            Half of the center-to-center well spacing, in microns.
        """
        return self.well_spacing_um / 2

    def generate_wells(self) -> List[PlateWell]:
        """
        Generate physical well center coordinates.

        Returns
        -------
        List[PlateWell]
            A list of physical well definitions, one per logical well on the plate,
            including logical identifiers and physical center coordinates.
        """
        plate_center = np.array([self.plate_width_mm, self.plate_height_mm]) * 1000 / 2
        a1_center = np.array(self.a1_offset_mm) * 1000 - plate_center

        wells: List[PlateWell] = []

        for row_index, row in enumerate(self.rows):
            for col_index, col in enumerate(self.cols):
                center = a1_center + np.array(
                    [
                        col_index * self.well_spacing_um,
                        row_index * self.well_spacing_um,
                    ]
                )
                wells.append(
                    PlateWell(
                        row=row,
                        col=col,
                        center_x=center[0],
                        center_y=center[1],
                    )
                )

        return wells


###############################################################################
# Plate Registry
###############################################################################

# Standard 96-well plate geometry.
PLATE_96 = Plate(
    name="96",
    rows=list("ABCDEFGH")[::-1],
    cols=[str(i) for i in range(1, 13)],
    plate_width_mm=126.6,
    plate_height_mm=85.7,
    a1_offset_mm=(14.3, 11.36),
    well_spacing_um=9000.0,
    well_radius_um=6210.0 / 2,
    assignment_mode=WellAssignmentMode.CLOSEST,
)


###############################################################################
# Position Extraction
###############################################################################


def extract_position_stage_xy_um(
    rdr: nd2.ND2File,
) -> Dict[int, Tuple[float, float]]:
    """
    Extract stage XY positions (µm) for each ND2 position index.

    Returns
    -------
    Dict[int, Tuple[float, float]]
        Mapping of ND2 position index → (x_um, y_um)

    """
    for exp in rdr.experiment:
        if "XYPosLoop" in str(exp):
            points = exp.parameters.points
            break
    else:
        raise RuntimeError("ND2 file does not contain XY position metadata.")

    return {
        i: (-p.stagePositionUm.x, -p.stagePositionUm.y) for i, p in enumerate(points)
    }


def extract_scene_to_position_index(
    rdr: nd2.ND2File,
    num_scenes: int,
) -> Dict[int, int]:
    """
    Map scene index → ND2 position index.

    Returns
    -------
    Dict[int, int]
        Mapping of absolute scene index to ND2 position index.
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
    *,
    plate: Plate,
) -> Optional[WellPosition]:
    """
    Assign a stage position to a logical well using the plate's assignment policy.

    Returns
    -------
    Optional[WellPosition]
        The assigned well, or None if the position does not satisfy the
        assignment criteria.
    """
    best = min(
        wells,
        key=lambda w: (x - w.center_x) ** 2 + (y - w.center_y) ** 2,
    )

    dist = np.sqrt((x - best.center_x) ** 2 + (y - best.center_y) ** 2)

    mode = plate.assignment_mode

    if mode is WellAssignmentMode.CLOSEST:
        return WellPosition(best.row, best.col)

    if mode is WellAssignmentMode.WITHIN_WELL:
        return (
            WellPosition(best.row, best.col) if dist <= plate.well_radius_um else None
        )

    if mode is WellAssignmentMode.HALF_SPACING:
        return (
            WellPosition(best.row, best.col) if dist <= plate.half_spacing_um else None
        )

    raise ValueError(f"Unknown WellAssignmentMode: {mode}")


def map_scenes_to_wells(
    scene_to_position: Dict[int, int],
    position_xy: Dict[int, Tuple[float, float]],
    wells: Iterable[PlateWell],
    *,
    plate: Plate,
) -> Dict[int, Optional[WellPosition]]:
    """
    Map absolute scene indices to logical well positions.

    This is the primary orchestration function used by the ND2 Reader.

    Returns
    -------
    Dict[int, Optional[WellPosition]]
        Mapping of absolute scene index to logical well position. If the
        plate's assignment policy rejects a position (e.g. strict physical
        containment), the value for that scene will be ``None``.
    """
    mapping: Dict[int, Optional[WellPosition]] = {}

    for scene_index, pos_index in scene_to_position.items():
        x, y = position_xy[pos_index]
        mapping[scene_index] = find_closest_well(
            x,
            y,
            wells,
            plate=plate,
        )

    return mapping
