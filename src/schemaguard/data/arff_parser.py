"""Controlled ARFF parsing with schema and row-width checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import arff
import pandas as pd

from .contracts import AttributeSchema


class ArffParseError(ValueError):
    """Raised when the raw ARFF cannot be parsed without ambiguity."""


@dataclass(frozen=True)
class ParsedArff:
    frame: pd.DataFrame
    relation: str
    attributes: tuple[AttributeSchema, ...]
    source_row_positions: tuple[int, ...]

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.attributes if item.kind == "numeric")

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.attributes if item.kind == "categorical")


def _attribute_kind(raw_type: Any) -> Literal["numeric", "categorical"]:
    normalized = str(raw_type).strip().lower()
    return "numeric" if normalized in {"numeric", "real", "integer", "number"} else "categorical"


def parse_arff(path: str | Path) -> ParsedArff:
    """Parse one ARFF file while preserving names, order, nulls, and row positions."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            parsed = arff.load(handle)
    except Exception as exc:  # liac-arff exposes several parser-specific exceptions.
        raise ArffParseError(f"Could not parse ARFF {source}: {exc}") from exc

    raw_attributes = parsed.get("attributes")
    raw_data = parsed.get("data")
    if not isinstance(raw_attributes, list) or not isinstance(raw_data, list):
        raise ArffParseError("ARFF is missing attributes or data")

    names = [str(item[0]) for item in raw_attributes]
    if len(names) != len(set(names)):
        raise ArffParseError("ARFF contains duplicate attribute names")

    attributes = tuple(
        AttributeSchema(
            name=name,
            raw_type=str(item[1]),
            kind=_attribute_kind(item[1]),
            is_target=False,
            position=position,
        )
        for position, (name, *_) in enumerate(raw_attributes)
        for item in [raw_attributes[position]]
    )

    expected_width = len(names)
    for position, row in enumerate(raw_data):
        if not isinstance(row, (list, tuple)) or len(row) != expected_width:
            observed_width = len(row) if isinstance(row, (list, tuple)) else "unknown"
            raise ArffParseError(
                f"ARFF row {position} has width {observed_width}; expected {expected_width}"
            )

    frame = pd.DataFrame(raw_data, columns=names)
    frame.insert(len(frame.columns), "__sg_source_row_position", range(len(frame)))
    return ParsedArff(
        frame=frame,
        relation=str(parsed.get("relation", "")),
        attributes=attributes,
        source_row_positions=tuple(range(len(frame))),
    )
