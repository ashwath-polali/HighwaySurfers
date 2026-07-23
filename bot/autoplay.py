"""UI navigator: detects the death screen and main menu, clicks back into a run.

Death screen: [Revive (19R$)] green  +  [Menu] orange   -> click MENU (orange)
Main menu:    [Play] green, centered                    -> click PLAY

Robux safety: a green button is only ever clicked when NO orange button is
visible AND it sits in the horizontal center band. "Revive" is green but
left-of-center and always appears next to the orange "Menu" — it can't match.

Buttons are found by color + shape, not position: solid saturated blob, wide
aspect, lower 60% of the screen, sane size, with white text inside. That last
check is what keeps grass (also green, also flat-shaded) out.
"""
import time

import cv2
import numpy as np


# HSV (OpenCV ranges: H 0..179)
GREEN_LO, GREEN_HI = (40, 110, 120), (85, 255, 255)
ORANGE_LO, ORANGE_HI = (8, 140, 140), (28, 255, 255)
WHITE_S_MAX, WHITE_V_MIN = 70, 200


class UINavigator:
    def __init__(self, region: tuple):
        self.region = region  # capture region (l, t, r, b) in screen coords
        self.last_click_t = 0.0
        self._streak_state = None
        self._streak = 0

    # ------------------------------------------------------------------
    def _buttons(self, hsv, white, lo, hi) -> list:
        """[(cx, cy, area), ...] of button-shaped blobs of the given color."""
        h, w = hsv.shape[:2]
        frame_area = h * w
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if not (0.002 * frame_area < area < 0.035 * frame_area):
                continue
            if bh == 0 or not (2.0 < bw / bh < 10.0):
                continue
            cy = y + bh / 2
            if cy < 0.50 * h:            # buttons live in the lower half
                continue
            if area / (bw * bh) < 0.55:  # solid rounded rect (minus its text)
                continue
            text_frac = white[y:y + bh, x:x + bw].mean() / 255.0
            if not (0.04 < text_frac < 0.55):
                continue
            out.append((float(centroids[i][0]), float(cy), int(area)))
        out.sort(key=lambda b: -b[2])
        return out

    def detect(self, frame) -> tuple:
        """-> (state, click_xy) where state in {'death', 'menu', None}."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_, v_ = hsv[:, :, 1], hsv[:, :, 2]
        white = ((s_ <= WHITE_S_MAX) & (v_ >= WHITE_V_MIN)).astype(np.uint8) * 255

        oranges = self._buttons(hsv, white, ORANGE_LO, ORANGE_HI)
        if oranges:  # death screen: always take Menu, never the green Revive
            return "death", (oranges[0][0], oranges[0][1])

        greens = self._buttons(hsv, white, GREEN_LO, GREEN_HI)
        w = frame.shape[1]
        for gx, gy, _a in greens:
            if 0.40 * w < gx < 0.60 * w:  # Play is centered; Revive is not
                return "menu", (gx, gy)
        return None, None

    # ------------------------------------------------------------------
    def step(self, frame, now: float, controls) -> str:
        """Detect + act. Returns 'death'/'menu' when UI is on screen ('' if not).

        Requires 2 consecutive detections before clicking so a single noisy
        frame can't trigger a stray click.
        """
        state, xy = self.detect(frame)
        if state is None:
            self._streak_state, self._streak = None, 0
            return ""

        if state == self._streak_state:
            self._streak += 1
        else:
            self._streak_state, self._streak = state, 1

        controls.release_all()  # never drive while a UI is up
        if self._streak >= 2 and now - self.last_click_t > 1.2:
            l, t, _r, _b = self.region
            self._click(int(l + xy[0]), int(t + xy[1]))
            self.last_click_t = now
            print(f"[autoplay] {state} screen -> clicked "
                  f"{'Menu' if state == 'death' else 'Play'}")
        return state

    @staticmethod
    def _click(x: int, y: int) -> None:
        import pydirectinput

        pydirectinput.moveTo(x, y)
        time.sleep(0.05)
        pydirectinput.click()
        # park the cursor low so it never hovers/occludes a button
        pydirectinput.moveTo(x, y + 60)
