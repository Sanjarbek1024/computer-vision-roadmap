"""Frame preprocessing (Phase 2: brightness, contrast, denoising).

Classroom cameras are usually mounted high, backlit by windows, and run in a
room that is half-lit during a projector session. That costs the detector real
recall on students sitting in the dark half of the room.

CLAHE on the L channel of LAB fixes local contrast without blowing out the
bright half of the frame the way a global brightness/contrast change would.

Note the split: detection runs on the *enhanced* frame, drawing happens on the
*original* one. The teacher should see the room as it looks, not as the
detector sees it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig | None = None):
        self.cfg = cfg or PreprocessConfig()
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip,
                tileGridSize=(self.cfg.clahe_grid, self.cfg.clahe_grid),
            )
            if self.cfg.clahe
            else None
        )

    @property
    def active(self) -> bool:
        return bool(self.cfg.clahe or self.cfg.denoise)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return a frame for the detector. The input is never modified."""
        if not self.active:
            return frame

        output = frame

        if self._clahe is not None:
            lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
            lightness, a, b = cv2.split(lab)
            lightness = self._clahe.apply(lightness)
            output = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)

        if self.cfg.denoise:
            ksize = self.cfg.denoise_ksize
            if ksize % 2 == 0:
                ksize += 1
            output = cv2.medianBlur(output, ksize)

        return output
