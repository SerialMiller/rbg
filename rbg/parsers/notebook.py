"""Parse Jupyter notebooks into auditable structures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CellOutput:
    """A single output from a notebook cell."""

    output_type: str  # stream, execute_result, display_data, error
    text: str = ""
    traceback: list[str] = field(default_factory=list)


@dataclass
class Cell:
    """A single notebook cell with its source, outputs, and metadata."""

    index: int
    cell_type: str  # code, markdown, raw
    source: str
    outputs: list[CellOutput] = field(default_factory=list)
    execution_count: Optional[int] = None

    @property
    def has_output(self) -> bool:
        return len(self.outputs) > 0

    @property
    def output_text(self) -> str:
        """Concatenate all text outputs."""
        parts = []
        for o in self.outputs:
            if o.text:
                parts.append(o.text)
        return "\n".join(parts)

    @property
    def has_error(self) -> bool:
        return any(o.output_type == "error" for o in self.outputs)

    @property
    def has_warning(self) -> bool:
        return "Warning" in self.output_text

    @property
    def convergence_warnings(self) -> list[str]:
        """Extract convergence warnings from outputs."""
        warnings = []
        text = self.output_text
        for line in text.split("\n"):
            if "ConvergenceWarning" in line or "did not converge" in line.lower():
                warnings.append(line.strip())
        return warnings


@dataclass
class Notebook:
    """Parsed Jupyter notebook."""

    path: Path
    kernel: str
    language: str
    cells: list[Cell]

    @property
    def code_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.cell_type == "code"]

    @property
    def markdown_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.cell_type == "markdown"]

    @property
    def all_code(self) -> str:
        """Concatenate all code cells."""
        return "\n\n".join(c.source for c in self.code_cells)

    @property
    def execution_order(self) -> list[Optional[int]]:
        """Return execution counts in cell order."""
        return [c.execution_count for c in self.code_cells]

    @property
    def has_out_of_order_execution(self) -> bool:
        """Check if cells were executed out of order."""
        counts = [c for c in self.execution_order if c is not None]
        if len(counts) < 2:
            return False
        return counts != sorted(counts)

    @property
    def execution_gaps(self) -> list[tuple[int, int, int]]:
        """Find gaps in execution counts that suggest hidden reruns.

        Returns list of (cell_index, prev_count, next_count) tuples.
        """
        gaps = []
        counts = [(c.index, c.execution_count) for c in self.code_cells
                   if c.execution_count is not None]
        for i in range(1, len(counts)):
            idx, curr = counts[i]
            _, prev = counts[i - 1]
            if curr - prev > 1:
                gaps.append((idx, prev, curr))
        return gaps

    @property
    def unexecuted_cells(self) -> list[Cell]:
        """Code cells with no execution count (never run or cleared)."""
        return [c for c in self.code_cells if c.execution_count is None]


def _extract_output(raw: dict) -> CellOutput:
    """Extract a CellOutput from a raw notebook output dict."""
    otype = raw.get("output_type", "unknown")
    text = ""
    traceback = []

    if otype in ("stream",):
        text = "".join(raw.get("text", []))
    elif otype in ("execute_result", "display_data"):
        data = raw.get("data", {})
        text = "".join(data.get("text/plain", []))
    elif otype == "error":
        traceback = raw.get("traceback", [])
        text = f"{raw.get('ename', 'Error')}: {raw.get('evalue', '')}"

    return CellOutput(output_type=otype, text=text, traceback=traceback)


def parse_notebook(path: str | Path) -> Notebook:
    """Parse a .ipynb file into a Notebook object."""
    path = Path(path)
    with open(path) as f:
        nb = json.load(f)

    # Extract kernel info
    kernelspec = nb.get("metadata", {}).get("kernelspec", {})
    kernel = kernelspec.get("display_name", "unknown")
    language = (
        nb.get("metadata", {})
        .get("language_info", {})
        .get("name", kernelspec.get("language", "unknown"))
    )

    cells = []
    for i, raw_cell in enumerate(nb.get("cells", [])):
        source = "".join(raw_cell.get("source", []))
        outputs = [
            _extract_output(o) for o in raw_cell.get("outputs", [])
        ]
        cell = Cell(
            index=i,
            cell_type=raw_cell.get("cell_type", "code"),
            source=source,
            outputs=outputs,
            execution_count=raw_cell.get("execution_count"),
        )
        cells.append(cell)

    return Notebook(path=path, kernel=kernel, language=language, cells=cells)
