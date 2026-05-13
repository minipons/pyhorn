from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ThroatAdapter:
    """Transition duct between the throat chamber and horn throat."""

    type: str = "cylindrical"
    ap1: float = 0.0
    lpt: float = 0.0


@dataclass
class RearChamber:
    """Rear chamber (sealed, vented, or coupling) parameters."""

    vrc: float = 0.0
    lrc: float = 0.0
    fr_rc: float = 0.0
    fr_tuning: float = 0.0
    width: float = 0.0
    height: float = 0.0
    depth: float = 0.0
    chamber_type: Literal["vented", "coupling", "sealed"] = "sealed"


@dataclass
class PassiveRadiator:
    """Passive radiator parameters."""

    mma: float = 0.0
    sp1: float = 0.0
    sp2: float = 0.0
    sp3: float = 0.0
    sp4: float = 0.0
    sp5: float = 0.0
    sp6: float = 0.0
    sp7: float = 0.0
    sp8: float = 0.0
    sp9: float = 0.0
    ql_pr: float = 5.0

    @property
    def total_sp(self) -> float:
        return (
            self.sp1
            + self.sp2
            + self.sp3
            + self.sp4
            + self.sp5
            + self.sp6
            + self.sp7
            + self.sp8
            + self.sp9
        )


@dataclass
class VentedBox:
    """Bass-reflex and hybrid vented-box parameters."""

    vrc: float = 0.0
    fr: float = 0.0
    lrc: float = 0.0
    ql: float = 5.0
    finite_horn_charged: bool = False
    path_length_difference: float = 0.0
    finite_transmission_line: bool = False
    ltl: float = 0.0


@dataclass
class SlavicBox:
    """Aperiodic slave-bass rear chamber."""

    vrc: float = 0.0
    rleak: float = 0.0
    aleak: float = 0.0
    lrc: float = 0.005


@dataclass
class ThroatChamber:
    """Sealed throat chamber at the horn throat."""

    vtc: float = 0.0
    atc: float = 0.0
    fr_tc: float = 0.0
