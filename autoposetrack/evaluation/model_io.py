"""Small dependency-free readers for BOP model geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_ascii_ply_vertices_m(path: Path) -> npt.NDArray[np.float64]:
    """Load vertex XYZ from an ASCII PLY whose units are millimetres."""

    with path.open("r", encoding="ascii") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError("not a PLY file")
        ascii_format = False
        vertex_count = None
        properties: list[str] = []
        in_vertex = False
        for line in handle:
            fields = line.strip().split()
            if fields[:2] == ["format", "ascii"]:
                ascii_format = True
                continue
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
                continue
            if fields[:1] == ["element"]:
                in_vertex = False
            if in_vertex and fields[:1] == ["property"]:
                properties.append(fields[-1])
            if fields[:1] == ["end_header"]:
                break
        if not ascii_format:
            raise ValueError("only ASCII PLY files are supported")
        if vertex_count is None or not {"x", "y", "z"}.issubset(properties):
            raise ValueError("PLY vertex header is incomplete")
        indices = [properties.index(axis) for axis in ("x", "y", "z")]
        vertices = []
        for _ in range(vertex_count):
            values = handle.readline().split()
            if len(values) < len(properties):
                raise ValueError("PLY ended before all vertices were read")
            vertices.append([float(values[index]) / 1000.0 for index in indices])
    return np.asarray(vertices, dtype=np.float64)
