"""Blender helpers shared by ``render.py`` and ``prepare_set.py``."""

from __future__ import annotations

import bpy
from mathutils import Matrix, Vector


def look_at(location: Vector, target: Vector, roll: float = 0.0) -> Matrix:
    rot = (target - location).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    return Matrix.Translation(location) @ rot @ Matrix.Rotation(roll, 4, "Z")


def setup_device(scene: bpy.types.Scene, device: str) -> str:
    scene.render.engine = "CYCLES"
    if device.upper() == "CPU":
        scene.cycles.device = "CPU"
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    backends = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"] if device == "auto" else [device.upper()]
    for backend in backends:
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return f"{backend}: " + ", ".join(d.name for d in gpus)
    raise SystemExit(f"Cycles found no GPU for {backends}; pass --device CPU to render on the CPU")
