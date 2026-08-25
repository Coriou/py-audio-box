"""Pose extraction backends for the auto-sfx POC.

Layout:

* :mod:`.mediapipe_backend` -- ``run_pose`` / ``ensure_models``
  (backends ``"mediapipe-lite"``, ``"mediapipe-full"``).
* :mod:`.rtmpose_backend`   -- same ``run_pose`` contract
  (backend ``"rtmpose-lightweight"``; comparison slice).

Shared cache helpers live in :mod:`auto_sfx.cache`. Heavy dependencies
(mediapipe, rtmlib, cv2) are imported lazily inside the backend modules, so
importing this package never pulls them in.
"""
