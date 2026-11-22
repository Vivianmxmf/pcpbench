from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

SOURCE_TRAIN, SOURCE_CAL, TARGET_TEST = 0, 1, 2


@dataclass(frozen=True)
class ShiftDataset:
    logits: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    domain: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return int(self.logits.shape[1])

    def _view(self, code: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.split == code
        return self.logits[m], self.labels[m]

    def source_cal(self) -> tuple[np.ndarray, np.ndarray]:
        return self._view(SOURCE_CAL)

    def target_test(self) -> tuple[np.ndarray, np.ndarray]:
        return self._view(TARGET_TEST)

    def to_npz(self, path) -> None:
        np.savez(
            Path(path),
            logits=self.logits.astype(np.float32),
            labels=self.labels.astype(np.int64),
            split=self.split.astype(np.int8),
            domain=self.domain.astype(np.int32),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def from_npz(cls, path) -> "ShiftDataset":
        z = np.load(Path(path), allow_pickle=False)
        return cls(
            logits=z["logits"],
            labels=z["labels"],
            split=z["split"],
            domain=z["domain"],
            meta=json.loads(str(z["meta"])),
        )
