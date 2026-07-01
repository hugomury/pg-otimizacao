"""I/O: IDs sequenciais, .dat, polares e CSV de log."""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import List, Sequence
import numpy as np


class DataLogger:
    def __init__(
        self,
        geometry_dir: Path,
        polar_dir: Path,
        csv_path: Path,
    ) -> None:
        self.geometry_dir = Path(geometry_dir)
        self.polar_dir = Path(polar_dir)
        self.csv_path = Path(csv_path)
        for d in (self.geometry_dir, self.polar_dir, self.csv_path.parent):
            d.mkdir(parents=True, exist_ok=True)

        self._counter = 0
        self._lock = threading.Lock()

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "ID",
                        "Algorithm",
                        "Generation",
                        "CST_Weights",
                        "Cl",
                        "Cd",
                        "Fitness",
                        "Status",
                    ]
                )

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"ID_{self._counter:04d}"

    @property
    def total_evaluations(self) -> int:
        return self._counter

    def save_geometry(self, airfoil_id: str, x: np.ndarray, y: np.ndarray) -> Path:
        path = self.geometry_dir / f"{airfoil_id}.dat"
        with open(path, "w") as f:
            f.write(f"{airfoil_id}\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi: .6f}  {yi: .6f}\n")
        return path

    def save_polar(self, airfoil_id: str, polar_data: Sequence) -> Path:
        path = self.polar_dir / f"{airfoil_id}_polar.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Alpha", "Cl", "Cd"])
            for row in polar_data:
                w.writerow(list(row))
        return path

    def log_evaluation(
        self,
        airfoil_id: str,
        algorithm: str,
        generation: int,
        cst_weights: List[float],
        cl: float,
        cd: float,
        fitness: float,
        status: str,
    ) -> None:
        with self._lock:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        airfoil_id,
                        algorithm,
                        generation,
                        # pesos serializados como string pra caber numa célula
                        ";".join(f"{v:.6f}" for v in cst_weights),
                        f"{cl:.6f}",
                        f"{cd:.6f}",
                        f"{fitness:.6f}",
                        status,
                    ]
                )
