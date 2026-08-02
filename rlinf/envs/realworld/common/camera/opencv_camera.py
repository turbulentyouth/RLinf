# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic USB camera capture via OpenCV's V4L2 backend.

Unlike :class:`LumosCamera` (which is hard-wired to the XVisio I420 stream),
this backend mirrors lerobot's ``OpenCVCamera``: it requests an MJPG stream
and lets OpenCV decode to BGR, so it works with ordinary UVC webcams such as
the Xense wrist cameras (``XC*`` serials) used by the Flexiv diagonal-02 rig.

Depth is not available from this interface.
"""

import glob
import os
from typing import Optional, Union

import numpy as np

from rlinf.utils.logging import get_logger

from .base_camera import BaseCamera, CameraInfo

_logger = get_logger()


class OpenCVCamera(BaseCamera):
    """Camera capture for generic UVC USB cameras (V4L2, MJPG stream).

    ``camera_info.serial_number`` may be:

    * a ``/dev/v4l/by-id/`` filename (preferred — stable across reboots)
    * a ``"videoN"`` shorthand resolved to ``/dev/videoN``
    * a numeric string or int interpreted as a V4L2 device index
    """

    def __init__(self, camera_info: CameraInfo):
        import cv2

        super().__init__(camera_info)
        self._cv2 = cv2

        if camera_info.enable_depth:
            raise ValueError("OpenCVCamera does not support depth capture via V4L2.")

        self._width, self._height = camera_info.resolution
        dev_path: Union[str, int] = self._resolve_device_path(camera_info.serial_number)

        self._cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open OpenCV camera (serial={camera_info.serial_number}, "
                f"dev_path={dev_path})."
            )

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, camera_info.fps)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:
            _logger.warning(
                "Failed to set OpenCV camera buffer size (serial=%s): %s",
                camera_info.serial_number,
                exc,
            )

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (self._width, self._height):
            _logger.warning(
                "OpenCV camera (serial=%s, dev_path=%s) negotiated %dx%d; "
                "requested %dx%d. Frames will be used as delivered.",
                camera_info.serial_number,
                dev_path,
                actual_w,
                actual_h,
                self._width,
                self._height,
            )

    @staticmethod
    def _resolve_device_path(serial_number: Union[str, int]) -> Union[str, int]:
        if isinstance(serial_number, int):
            return serial_number
        if serial_number.startswith("/dev/"):
            return serial_number
        if serial_number.startswith("video"):
            return f"/dev/{serial_number}"
        by_id = f"/dev/v4l/by-id/{serial_number}"
        if os.path.exists(by_id):
            return by_id
        try:
            return int(serial_number)
        except ValueError as exc:
            raise ValueError(
                f"Could not resolve OpenCV camera serial_number={serial_number!r} "
                "to a V4L2 device. Use a /dev/v4l/by-id/ name, videoN, or an index."
            ) from exc

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def _close_device(self) -> None:
        if self._cap is not None:
            self._cap.release()

    @staticmethod
    def get_device_serial_numbers() -> list[str]:
        """Return stable by-id identifiers for connected V4L2 cameras.

        Falls back to ``videoN`` names when ``/dev/v4l/by-id/`` is unavailable.
        """
        devices = glob.glob("/dev/v4l/by-id/*")
        if devices:
            return [os.path.basename(d) for d in devices]
        return [os.path.basename(v) for v in glob.glob("/dev/video*")]
