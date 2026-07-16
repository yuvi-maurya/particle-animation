from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ----------------------------- Tunable constants -----------------------------

CAMERA_INDEX = 0
WINDOW_NAME = "3D Model Box"

EXPECTED_MODELS = ("dragon.obj", "tree.obj", "flowers.obj", "butterfly.obj")
MODEL_SEARCH_DIRS = (Path("assets"), Path("assets") / "models")
MODEL_FIT_RATIO = 0.68
TARGET_RENDER_TRIANGLES = 3000
MODEL_CACHE_SCHEMA_VERSION = "obj-render-cache-v2"
ORIENTATION_OUTPUT_DIR = Path("orientation_renders")
MIN_PROJECTED_TRIANGLE_AREA = 0.02
TEXTURE_ALPHA_SKIP_THRESHOLD = 200
TEXTURE_ALPHA_OPAQUE_THRESHOLD = 224

MAX_NUM_HANDS = 2
MEDIAPIPE_EVERY_N_FRAMES = 2
SWAP_HANDEDNESS = False
HAND_LOST_GRACE_FRAMES = 7
SMOOTHING_ALPHA = 0.14

MIN_CUBE_SIZE_PX = 250.0
MAX_CUBE_SIZE_PX = 560.0
CUBE_FOCAL_LENGTH = 540.0
CUBE_CAMERA_DISTANCE_FACTOR = 3.45
CUBE_LINE_COLOR = (0, 230, 255)
CUBE_REAR_LINE_COLOR = (0, 100, 130)
CUBE_LINE_THICKNESS = 2
CUBE_FRONT_ALPHA = 0.58
CUBE_REAR_ALPHA = 0.28

Y_ROTATION_RADIANS_PER_SECOND = math.radians(90.0)
FIXED_X_TILT = 0.0
MAX_DELTA_TIME = 0.05

RIGHT_FINGER_EXTENDED_DEGREES = 155.0
RIGHT_FINGER_FOLDED_DEGREES = 120.0
RIGHT_FIST_STABLE_FRAMES = 3
RIGHT_FIST_COOLDOWN_SECONDS = 0.42
MODEL_TRANSITION_SECONDS = 0.85
SMOKE_PARTICLE_COUNT = 64
LEFT_EMERGE_SECONDS = 0.55

LEFT_HIDE_RATIO = 0.30
LEFT_SHOW_RATIO = 0.85
LEFT_VISIBILITY_STABLE_FRAMES = 2

MAX_CAMERA_READ_FAILURES = 12
CAMERA_WARMUP_FRAMES = 8
CAMERA_VALIDATION_FRAMES = 10
CAMERA_START_TIMEOUT_SECONDS = 5.0
CAMERA_STALE_SECONDS = 1.0
CAMERA_OUTPUT_SIZE = (640, 480)
DEFAULT_SHOW_DEBUG_INFO = False
DEFAULT_SHOW_LANDMARKS = False


# ----------------------------- Model loading ---------------------------------


@dataclass
class ModelConfig:
    filename: str
    rotation_x_deg: float = 0.0
    rotation_y_deg: float = 0.0
    rotation_z_deg: float = 0.0
    scale_multiplier: float = 1.0
    vertical_offset: float = 0.0
    fallback_bgr: tuple[int, int, int] = (170, 170, 190)


MODEL_CONFIGS = {
    "dragon.obj": ModelConfig("dragon.obj", fallback_bgr=(95, 140, 210)),
    "tree.obj": ModelConfig("tree.obj", fallback_bgr=(70, 155, 90)),
    "flowers.obj": ModelConfig("flowers.obj", fallback_bgr=(190, 120, 210)),
    "butterfly.obj": ModelConfig("butterfly.obj", fallback_bgr=(220, 155, 65)),
}


@dataclass
class ObjModel:
    name: str
    path: Path
    vertices: np.ndarray
    faces: np.ndarray
    face_colors: np.ndarray
    face_alphas: np.ndarray
    face_materials: list[str]
    material_names: list[str]
    texture_dependencies: list[str]
    source_fingerprint: str
    source_up_axis: str
    front_normal_sign: float
    bounds: tuple[np.ndarray, np.ndarray]
    normalized_bounds: tuple[np.ndarray, np.ndarray]
    face_normals: np.ndarray
    original_triangle_count: int
    render_triangle_count: int
    orientation_degrees: tuple[float, float, float]
    load_error: str | None = None


@dataclass
class MtlInfo:
    material_colors: dict[str, tuple[int, int, int]]
    material_textures: dict[str, Path]
    texture_dependencies: list[Path]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_case_insensitive(path: Path) -> Path:
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return path
    target = path.name.lower()
    for candidate in parent.iterdir():
        if candidate.name.lower() == target:
            return candidate
    return path


def config_fingerprint(config: ModelConfig) -> str:
    return json.dumps(config.__dict__, sort_keys=True)


def parse_mtl(mtl_path: Path) -> MtlInfo:
    materials: dict[str, tuple[int, int, int]] = {}
    material_textures: dict[str, Path] = {}
    textures: list[Path] = []
    current: str | None = None
    if not mtl_path.exists():
        return MtlInfo(materials, material_textures, textures)

    for raw in mtl_path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].lower()
        if key == "newmtl" and len(parts) > 1:
            current = parts[1]
        elif key == "kd" and current and len(parts) >= 4:
            rgb = [float(parts[1]), float(parts[2]), float(parts[3])]
            materials[current] = tuple(int(np.clip(c, 0.0, 1.0) * 255) for c in rgb[::-1])
        elif key.startswith("map_") and current and len(parts) > 1:
            tex_path = resolve_case_insensitive(mtl_path.parent / parts[-1].replace("\\", "/"))
            textures.append(tex_path)
            if key in {"map_kd", "map_d"} and current not in material_textures:
                material_textures[current] = tex_path
    return MtlInfo(materials, material_textures, sorted(set(textures)))


def discover_model_path(root: Path, filename: str) -> tuple[Path | None, list[Path]]:
    key = Path(filename).stem.lower()
    candidates: list[Path] = []
    aliases = {key}
    if key == "flowers":
        aliases.add("flower")
    if key == "butterfly":
        aliases.add("butterflly")
    for folder in MODEL_SEARCH_DIRS:
        directory = root / folder
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() != ".obj":
                continue
            stem = path.stem.lower()
            if any(alias in stem for alias in aliases):
                candidates.append(path)
    if not candidates:
        return None, []
    # The user-updated current workspace file wins. Newer source file wins over stale duplicates.
    return max(candidates, key=lambda p: p.stat().st_mtime), sorted(candidates)


def model_source_fingerprint(path: Path, config: ModelConfig, mtllibs: list[Path], textures: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(MODEL_CACHE_SCHEMA_VERSION.encode("utf-8"))
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(file_sha256(path).encode("utf-8"))
    for dep in sorted(mtllibs + textures, key=lambda p: str(p).lower()):
        digest.update(str(dep.resolve()).encode("utf-8"))
        digest.update(b"exists" if dep.exists() else b"missing")
        if dep.exists():
            digest.update(file_sha256(dep).encode("utf-8"))
    digest.update(config_fingerprint(config).encode("utf-8"))
    return digest.hexdigest()


def face_index(token: str, vertex_count: int) -> int:
    raw = int(token.split("/")[0])
    return raw - 1 if raw > 0 else vertex_count + raw


def create_rotation_matrix_xyz(x: float, y: float, z: float) -> np.ndarray:
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(lengths, 1e-8)).astype(np.float32)


def estimate_front_normal_sign(normals: np.ndarray) -> float:
    z = normals[:, 2]
    nonzero = z[np.abs(z) > 1e-5]
    if len(nonzero) == 0:
        return -1.0
    return -1.0 if np.count_nonzero(nonzero < 0) >= np.count_nonzero(nonzero > 0) else 1.0


def load_obj_builtin(path: Path, config: ModelConfig) -> ObjModel:
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals_seen = 0
    triangles: list[list[int]] = []
    triangle_uvs: list[list[int | None]] = []
    face_materials: list[str] = []
    mtllibs: list[Path] = []
    current_material = "default"

    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "vt" and len(parts) >= 3:
            texcoords.append([float(parts[1]), float(parts[2])])
        elif parts[0] == "vn" and len(parts) >= 4:
            normals_seen += 1
        elif parts[0] == "mtllib" and len(parts) > 1:
            mtllibs.append(resolve_case_insensitive(path.parent / " ".join(parts[1:]).replace("\\", "/")))
        elif parts[0] == "usemtl" and len(parts) > 1:
            current_material = " ".join(parts[1:])
        elif parts[0] == "f" and len(parts) >= 4:
            idx = [face_index(token, len(vertices)) for token in parts[1:]]
            uv_idx: list[int | None] = []
            for token in parts[1:]:
                fields = token.split("/")
                if len(fields) >= 2 and fields[1]:
                    raw_uv = int(fields[1])
                    uv_idx.append(raw_uv - 1 if raw_uv > 0 else len(texcoords) + raw_uv)
                else:
                    uv_idx.append(None)
            for i in range(1, len(idx) - 1):
                triangles.append([idx[0], idx[i], idx[i + 1]])
                triangle_uvs.append([uv_idx[0], uv_idx[i], uv_idx[i + 1]])
                face_materials.append(current_material)

    if not vertices:
        raise ValueError("no vertices")
    if not triangles:
        raise ValueError("no faces")

    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(triangles, dtype=np.int32)
    if not np.isfinite(vertices_np).all():
        raise ValueError("non-finite vertices")
    if faces_np.min() < 0 or faces_np.max() >= len(vertices_np):
        raise ValueError("face index out of range")

    material_colors: dict[str, tuple[int, int, int]] = {"default": config.fallback_bgr}
    material_textures: dict[str, Path] = {}
    textures: list[Path] = []
    for mtl in mtllibs:
        if not mtl.exists():
            print(f"Model package warning: {path.name} references missing MTL {mtl}")
            continue
        info = parse_mtl(mtl)
        material_colors.update(info.material_colors)
        material_textures.update(info.material_textures)
        textures.extend(info.texture_dependencies)

    decoded_textures: dict[Path, np.ndarray] = {}
    for texture_path in sorted(set(textures)):
        if not texture_path.exists():
            print(f"Model package warning: {path.name} references missing texture {texture_path}")
            continue
        decoded = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            print(f"Model package warning: {path.name} texture could not decode {texture_path}")
            continue
        decoded_textures[texture_path] = decoded

    colors_list: list[tuple[int, int, int]] = []
    alpha_list: list[int] = []
    texcoords_np = np.asarray(texcoords, dtype=np.float32) if texcoords else np.empty((0, 2), dtype=np.float32)
    for material_name, tri_uv in zip(face_materials, triangle_uvs):
        texture_path = material_textures.get(material_name)
        sampled: tuple[int, int, int] | None = None
        alpha = 255
        if texture_path in decoded_textures and all(idx is not None and 0 <= idx < len(texcoords_np) for idx in tri_uv):
            uv = texcoords_np[[int(idx) for idx in tri_uv]].mean(axis=0)
            texture = decoded_textures[texture_path]
            h, w = texture.shape[:2]
            x = int(np.clip(uv[0], 0.0, 1.0) * (w - 1))
            y = int((1.0 - np.clip(uv[1], 0.0, 1.0)) * (h - 1))
            pixel = texture[y, x]
            sampled = tuple(int(v) for v in pixel[:3])
            if texture.shape[2] >= 4:
                alpha = int(pixel[3])
                if alpha < TEXTURE_ALPHA_SKIP_THRESHOLD:
                    alpha = 0
                elif alpha >= TEXTURE_ALPHA_OPAQUE_THRESHOLD:
                    alpha = 255
        colors_list.append(sampled or material_colors.get(material_name, config.fallback_bgr))
        alpha_list.append(alpha)

    colors = np.array(colors_list, dtype=np.uint8)
    alphas = np.array(alpha_list, dtype=np.uint8)
    fingerprint = model_source_fingerprint(path, config, mtllibs, sorted(set(textures)))
    print(
        f"{path.name} - vertices: {len(vertices_np)}, triangles: {len(faces_np)}, "
        f"MTL: {'found' if any(m.exists() for m in mtllibs) else 'missing'}, "
        f"vt: {len(texcoords)}, vn: {normals_seen}, "
        f"textures: {[str(t) for t in sorted(set(textures))]}"
    )
    print(f"Cache rebuilt: {path.name} - source fingerprint changed ({fingerprint[:12]})")
    return normalize_model(
        ObjModel(
            name=path.name,
            path=path,
            vertices=vertices_np,
            faces=faces_np,
            face_colors=colors,
            face_alphas=alphas,
            face_materials=face_materials,
            material_names=sorted(set(face_materials)),
            texture_dependencies=[str(t) for t in sorted(set(textures))],
            source_fingerprint=fingerprint,
            source_up_axis="Y-up",
            front_normal_sign=estimate_front_normal_sign(compute_face_normals(vertices_np, faces_np)),
            bounds=(vertices_np.min(axis=0), vertices_np.max(axis=0)),
            normalized_bounds=(vertices_np.min(axis=0), vertices_np.max(axis=0)),
            face_normals=compute_face_normals(vertices_np, faces_np),
            original_triangle_count=len(faces_np),
            render_triangle_count=len(faces_np),
            orientation_degrees=(config.rotation_x_deg, config.rotation_y_deg, config.rotation_z_deg),
        ),
        config,
    )


def load_obj_trimesh(path: Path, config: ModelConfig) -> ObjModel:
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if len(geom.vertices) and len(geom.faces)]
        if not geometries:
            raise ValueError("trimesh scene contained no usable geometries")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(vertices) == 0:
        raise ValueError("trimesh found no vertices")
    if len(faces) == 0:
        raise ValueError("trimesh found no faces")
    if faces.shape[1] != 3:
        mesh = mesh.triangulate()
        faces = np.asarray(mesh.faces, dtype=np.int32)

    face_materials = ["trimesh_material"] * len(faces)
    face_colors = np.tile(np.array(config.fallback_bgr, dtype=np.uint8), (len(faces), 1))
    face_alphas = np.full(len(faces), 255, dtype=np.uint8)
    visual_colors = getattr(mesh.visual, "face_colors", None)
    if visual_colors is not None and len(visual_colors) == len(faces):
        rgb = np.asarray(visual_colors[:, :3], dtype=np.uint8)
        face_colors = rgb[:, ::-1].copy()
        if visual_colors.shape[1] >= 4:
            face_alphas = np.asarray(visual_colors[:, 3], dtype=np.uint8).copy()

    textures: list[Path] = []
    for mtl_path in path.parent.glob("*.mtl"):
        info = parse_mtl(mtl_path)
        textures.extend(info.texture_dependencies)
    fingerprint = model_source_fingerprint(path, config, sorted(path.parent.glob("*.mtl")), sorted(set(textures)))

    return normalize_model(
        ObjModel(
            name=path.name,
            path=path,
            vertices=vertices,
            faces=faces,
            face_colors=face_colors,
            face_alphas=face_alphas,
            face_materials=face_materials,
            material_names=sorted(set(face_materials)),
            texture_dependencies=[str(t) for t in sorted(set(textures))],
            source_fingerprint=fingerprint,
            source_up_axis="Y-up",
            front_normal_sign=estimate_front_normal_sign(compute_face_normals(vertices, faces)),
            bounds=(vertices.min(axis=0), vertices.max(axis=0)),
            normalized_bounds=(vertices.min(axis=0), vertices.max(axis=0)),
            face_normals=compute_face_normals(vertices, faces),
            original_triangle_count=len(faces),
            render_triangle_count=len(faces),
            orientation_degrees=(config.rotation_x_deg, config.rotation_y_deg, config.rotation_z_deg),
        ),
        config,
    )


def normalize_model(model: ObjModel, config: ModelConfig) -> ObjModel:
    vertices = model.vertices.astype(np.float32).copy()
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    vertices -= center
    dims = vertices.max(axis=0) - vertices.min(axis=0)
    largest = float(np.max(dims))
    if largest <= 1e-8 or not np.isfinite(largest):
        raise ValueError("zero or invalid model size")

    vertices *= (MODEL_FIT_RATIO * config.scale_multiplier) / largest
    rotation = create_rotation_matrix_xyz(
        math.radians(config.rotation_x_deg),
        math.radians(config.rotation_y_deg),
        math.radians(config.rotation_z_deg),
    )
    vertices = vertices @ rotation.T
    vertices[:, 1] += config.vertical_offset
    normals = compute_face_normals(vertices, model.faces)

    return ObjModel(
        name=model.name,
        path=model.path,
        vertices=vertices.astype(np.float32),
        faces=model.faces.astype(np.int32),
        face_colors=model.face_colors.astype(np.uint8),
        face_alphas=model.face_alphas.astype(np.uint8),
        face_materials=model.face_materials,
        material_names=model.material_names,
        texture_dependencies=model.texture_dependencies,
        source_fingerprint=model.source_fingerprint,
        source_up_axis=model.source_up_axis,
        front_normal_sign=model.front_normal_sign,
        bounds=model.bounds,
        normalized_bounds=(vertices.min(axis=0), vertices.max(axis=0)),
        face_normals=normals,
        original_triangle_count=model.original_triangle_count,
        render_triangle_count=len(model.faces),
        orientation_degrees=(config.rotation_x_deg, config.rotation_y_deg, config.rotation_z_deg),
    )


def maybe_decimate(model: ObjModel) -> ObjModel:
    if len(model.faces) <= TARGET_RENDER_TRIANGLES:
        return model
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=model.vertices, faces=model.faces, process=False)
        simplified = mesh.simplify_quadric_decimation(face_count=TARGET_RENDER_TRIANGLES)
        faces = np.asarray(simplified.faces, dtype=np.int32)
        vertices = np.asarray(simplified.vertices, dtype=np.float32)
        print(f"{model.name}: original {model.original_triangle_count} triangles -> render LOD {len(faces)} triangles")
        return ObjModel(
            name=model.name,
            path=model.path,
            vertices=vertices,
            faces=faces,
            face_colors=np.tile(model.face_colors[0], (len(faces), 1)).astype(np.uint8),
            face_alphas=np.full(len(faces), int(model.face_alphas[0]), dtype=np.uint8),
            face_materials=["lod_material"] * len(faces),
            material_names=model.material_names,
            texture_dependencies=model.texture_dependencies,
            source_fingerprint=model.source_fingerprint,
            source_up_axis=model.source_up_axis,
            front_normal_sign=model.front_normal_sign,
            bounds=model.bounds,
            normalized_bounds=(vertices.min(axis=0), vertices.max(axis=0)),
            face_normals=compute_face_normals(vertices, faces),
            original_triangle_count=model.original_triangle_count,
            render_triangle_count=len(faces),
            orientation_degrees=model.orientation_degrees,
        )
    except Exception as exc:
        print(f"{model.name}: quadric simplification unavailable/failed ({exc}); using vertex-cluster LOD")
    clustered = build_vertex_cluster_lod(model, TARGET_RENDER_TRIANGLES)
    print(f"{model.name}: original {model.original_triangle_count} triangles -> render LOD {len(clustered.faces)} triangles")
    return clustered


def build_vertex_cluster_lod(model: ObjModel, target_faces: int) -> ObjModel:
    bounds_min = model.vertices.min(axis=0)
    bounds_max = model.vertices.max(axis=0)
    span = np.maximum(bounds_max - bounds_min, 1e-6)
    best: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]] | None = None
    best_score = float("inf")

    for resolution in (10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 64):
        scaled = np.floor((model.vertices - bounds_min) / span * resolution).astype(np.int32)
        _, inverse = np.unique(scaled, axis=0, return_inverse=True)
        remapped = inverse[model.faces]
        valid = (
            (remapped[:, 0] != remapped[:, 1])
            & (remapped[:, 1] != remapped[:, 2])
            & (remapped[:, 0] != remapped[:, 2])
        )
        if not np.any(valid):
            continue
        faces = remapped[valid]
        _, unique_face_indices = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        unique_face_indices = np.sort(unique_face_indices)
        faces = faces[unique_face_indices]
        source_indices = np.flatnonzero(valid)[unique_face_indices]

        vertex_count = int(inverse.max()) + 1
        new_vertices = np.zeros((vertex_count, 3), dtype=np.float32)
        counts = np.bincount(inverse, minlength=vertex_count).astype(np.float32)
        for axis in range(3):
            new_vertices[:, axis] = np.bincount(inverse, weights=model.vertices[:, axis], minlength=vertex_count)
        new_vertices /= np.maximum(counts[:, None], 1.0)

        score = abs(len(faces) - target_faces)
        if len(faces) <= target_faces:
            score *= 0.5
        if score < best_score:
            best_score = score
            best = (
                new_vertices,
                faces.astype(np.int32),
                model.face_colors[source_indices],
                model.face_alphas[source_indices],
                [model.face_materials[i] for i in source_indices],
            )

    if best is None:
        return model

    vertices, faces, colors, alphas, materials = best
    return ObjModel(
        name=model.name,
        path=model.path,
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        face_colors=colors.astype(np.uint8),
        face_alphas=alphas.astype(np.uint8),
        face_materials=materials,
        material_names=model.material_names,
        texture_dependencies=model.texture_dependencies,
        source_fingerprint=model.source_fingerprint,
        source_up_axis=model.source_up_axis,
        front_normal_sign=estimate_front_normal_sign(compute_face_normals(vertices, faces)),
        bounds=model.bounds,
        normalized_bounds=(vertices.min(axis=0), vertices.max(axis=0)),
        face_normals=compute_face_normals(vertices, faces),
        original_triangle_count=model.original_triangle_count,
        render_triangle_count=len(faces),
        orientation_degrees=model.orientation_degrees,
    )


class ModelManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.models: list[ObjModel] = []
        self.invalid: list[tuple[Path, str]] = []
        self.candidate_decisions: dict[str, list[Path]] = {}
        self.index = 0
        self.load_expected_models()

    @property
    def current(self) -> ObjModel | None:
        return self.models[self.index] if self.models else None

    def load_expected_models(self) -> None:
        for filename in EXPECTED_MODELS:
            path, candidates = discover_model_path(self.root, filename)
            self.candidate_decisions[filename] = candidates
            config = MODEL_CONFIGS[filename]
            if path is None:
                expected = self.root / MODEL_SEARCH_DIRS[0] / filename
                self.invalid.append((expected, "missing"))
                print(f"Model load error: {expected} -> missing")
                continue
            if len(candidates) > 1:
                print(f"Model candidates for {filename}: {[str(p) for p in candidates]}")
                print(f"Selected newest source: {path}")
            try:
                try:
                    # The built-in loader preserves usemtl/vt records and bakes texture colours.
                    model = load_obj_builtin(path, config)
                    print(f"OBJ loader: built-in package-aware loader -> {path.name}")
                except Exception:
                    raise
                model = maybe_decimate(model)
                self.models.append(model)
                print_model_report(model)
            except Exception as exc:
                self.invalid.append((path, str(exc)))
                print(f"Model load error: {path} -> {exc}")
        if not self.models:
            print("No valid OBJ models loaded. The app will run, but no model will render.")

    def change(self, delta: int) -> tuple[ObjModel, ObjModel] | None:
        if not self.models:
            print("Model change ignored: no valid OBJ models loaded.")
            return None
        old_index = self.index
        old_name = self.current.name
        old_model = self.current
        self.index = (self.index + delta) % len(self.models)
        print(f"Model changed: {old_index + 1}/{len(self.models)} {old_name} -> {self.index + 1}/{len(self.models)} {self.current.name}")
        return old_model, self.current

    def next(self) -> tuple[ObjModel, ObjModel] | None:
        return self.change(1)

    def previous(self) -> tuple[ObjModel, ObjModel] | None:
        return self.change(-1)


def print_model_report(model: ObjModel) -> None:
    print(f"Model loaded: {model.path}")
    print(f"  vertices: {len(model.vertices)}")
    print(f"  triangles: original {model.original_triangle_count}, render {model.render_triangle_count}")
    print(f"  materials: {', '.join(model.material_names) if model.material_names else 'none'}")
    print(f"  texture dependencies: {', '.join(model.texture_dependencies) if model.texture_dependencies else 'none'}")
    print(f"  normalized bounds: min={np.round(model.normalized_bounds[0], 4).tolist()} max={np.round(model.normalized_bounds[1], 4).tolist()}")
    print(f"  orientation correction xyz degrees: {model.orientation_degrees}")


# ----------------------------- Hand tracking ---------------------------------


class HandRoleResolver:
    def __init__(self, swap_handedness: bool = SWAP_HANDEDNESS) -> None:
        self.swap_handedness = swap_handedness

    def toggle(self) -> None:
        self.swap_handedness = not self.swap_handedness
        print(f"Hand mapping toggled: {'SWAPPED' if self.swap_handedness else 'NORMAL'}")

    def resolve_physical_hand_label(self, mp_label: str) -> str:
        if self.swap_handedness:
            return "Left" if mp_label == "Right" else "Right"
        return mp_label

    def resolve(self, results) -> list[dict]:
        if results is None or not results.multi_hand_landmarks or not results.multi_handedness:
            return []
        resolved = []
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            item = handedness.classification[0]
            resolved.append(
                {
                    "landmarks": landmarks,
                    "mp_label": item.label,
                    "physical_label": self.resolve_physical_hand_label(item.label),
                    "score": item.score,
                }
            )
        return resolved


class HandTracker:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )

    def process(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.hands.process(rgb)

    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> None:
        if results is None or not results.multi_hand_landmarks:
            return
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

    def close(self) -> None:
        self.hands.close()


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([(lm.x * width, lm.y * height) for lm in landmarks.landmark], dtype=np.float32)


@dataclass
class ObjectPose:
    anchor: tuple[int, int]
    cube_size: float
    target_size: float


class LeftHandPoseSmoother:
    def __init__(self, alpha: float = SMOOTHING_ALPHA) -> None:
        self.alpha = alpha
        self.anchor: np.ndarray | None = None
        self.size: float | None = None
        self.last_pose: ObjectPose | None = None
        self.missing_frames = HAND_LOST_GRACE_FRAMES + 1

    def update(self, landmarks, frame_shape: tuple[int, int, int], scale: float = 1.0) -> ObjectPose | None:
        if landmarks is None:
            self.missing_frames += 1
            if self.missing_frames <= HAND_LOST_GRACE_FRAMES:
                return self.last_pose
            self.anchor = None
            self.size = None
            self.last_pose = None
            return None

        self.missing_frames = 0
        height, width = frame_shape[:2]
        points = landmarks_to_pixels(landmarks, width, height)
        raw_finger_midpoint = (points[4] + points[8]) * 0.5
        palm_width = float(np.linalg.norm(points[5] - points[17]))
        palm_height = float(np.linalg.norm(points[0] - points[9]))
        raw_size = float(np.clip(max(palm_width * 4.5, palm_height * 4.8), MIN_CUBE_SIZE_PX, MAX_CUBE_SIZE_PX))
        raw_anchor = raw_finger_midpoint

        if self.anchor is None:
            self.anchor = raw_anchor
            self.size = raw_size
        else:
            self.anchor = self.anchor * (1.0 - self.alpha) + raw_anchor * self.alpha
            self.size = self.size * (1.0 - self.alpha) + raw_size * self.alpha

        target_size = float(self.size)
        current_size = float(target_size * np.clip(scale, 0.05, 1.0))
        self.last_pose = ObjectPose((int(self.anchor[0]), int(self.anchor[1])), current_size, target_size)
        return self.last_pose


# ----------------------------- Gesture controls ------------------------------


@dataclass
class RightFistDiagnostics:
    gesture: str = "UNKNOWN"
    stable_state: str = "UNKNOWN"
    extended_count: int = 0
    folded_count: int = 0
    armed: bool = False
    cooldown_remaining: float = 0.0


@dataclass
class LeftVisibilityDiagnostics:
    ratio: float | None = None
    raw_state: str = "UNKNOWN"
    stable_state: str = "WIDE"
    visible: bool = True


def thumb_index_ratio(landmarks, frame_shape: tuple[int, int, int]) -> float:
    height, width = frame_shape[:2]
    points = landmarks_to_pixels(landmarks, width, height)
    pinch_distance = float(np.linalg.norm(points[4] - points[8]))
    palm_size = float(np.linalg.norm(points[5] - points[17]))
    return pinch_distance / max(palm_size, 1e-6)


def joint_angle_degrees(points: np.ndarray, mcp: int, pip: int, tip: int) -> float:
    a = points[mcp] - points[pip]
    b = points[tip] - points[pip]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def classify_right_open_fist(landmarks, frame_shape: tuple[int, int, int]) -> tuple[str, int, int, list[float]]:
    height, width = frame_shape[:2]
    points = landmarks_to_pixels(landmarks, width, height)
    fingers = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    angles = [joint_angle_degrees(points, *finger) for finger in fingers]
    extended = sum(angle >= RIGHT_FINGER_EXTENDED_DEGREES for angle in angles)
    folded = sum(angle <= RIGHT_FINGER_FOLDED_DEGREES for angle in angles)
    if extended >= 3:
        return "OPEN", int(extended), int(folded), angles
    if folded >= 3:
        return "CLOSED", int(extended), int(folded), angles
    return "NEUTRAL", int(extended), int(folded), angles


class RightFistStateMachine:
    def __init__(self) -> None:
        self.raw_state = "UNKNOWN"
        self.stable_state = "UNKNOWN"
        self.armed = True
        self.recent: deque[str] = deque(maxlen=RIGHT_FIST_STABLE_FRAMES)
        self.last_switch_time = -float("inf")
        self.extended_count = 0
        self.folded_count = 0

    def update(self, gesture: str | None, now: float, extended_count: int = 0, folded_count: int = 0) -> tuple[bool, RightFistDiagnostics]:
        if gesture is not None:
            self.raw_state = gesture
            self.extended_count = extended_count
            self.folded_count = folded_count
        self.recent.append(self.raw_state)
        changed = False

        if len(self.recent) == RIGHT_FIST_STABLE_FRAMES and len(set(self.recent)) == 1:
            candidate = self.recent[0]
            if candidate != self.stable_state:
                self.stable_state = candidate
                if candidate == "CLOSED":
                    self.armed = True
                elif candidate == "OPEN" and self.armed and now - self.last_switch_time >= RIGHT_FIST_COOLDOWN_SECONDS:
                    changed = True
                    self.armed = False
                    self.last_switch_time = now

        cooldown = max(0.0, RIGHT_FIST_COOLDOWN_SECONDS - (now - self.last_switch_time))
        return changed, RightFistDiagnostics(
            gesture=self.raw_state,
            stable_state=self.stable_state,
            extended_count=self.extended_count,
            folded_count=self.folded_count,
            armed=self.armed,
            cooldown_remaining=cooldown,
        )

    def diagnostics(self, now: float) -> RightFistDiagnostics:
        return RightFistDiagnostics(
            gesture=self.raw_state,
            stable_state=self.stable_state,
            extended_count=self.extended_count,
            folded_count=self.folded_count,
            armed=self.armed,
            cooldown_remaining=max(0.0, RIGHT_FIST_COOLDOWN_SECONDS - (now - self.last_switch_time)),
        )


class LeftVisibilityStateMachine:
    def __init__(self) -> None:
        self.raw_state = "WIDE"
        self.stable_state = "WIDE"
        self.object_visible_latch = True
        self.recent: deque[str] = deque(maxlen=LEFT_VISIBILITY_STABLE_FRAMES)
        self.just_became_visible = False

    def classify(self, ratio: float | None) -> str:
        if ratio is None:
            return self.raw_state
        if ratio <= LEFT_HIDE_RATIO:
            return "PINCHED"
        if ratio >= LEFT_SHOW_RATIO:
            return "WIDE"
        return self.raw_state

    def update(self, ratio: float | None) -> LeftVisibilityDiagnostics:
        previous_visible = self.object_visible_latch
        self.just_became_visible = False
        self.raw_state = self.classify(ratio)
        self.recent.append(self.raw_state)
        if len(self.recent) == LEFT_VISIBILITY_STABLE_FRAMES and len(set(self.recent)) == 1:
            self.stable_state = self.recent[0]
            if self.stable_state == "PINCHED":
                self.object_visible_latch = False
            elif self.stable_state == "WIDE":
                self.object_visible_latch = True
        self.just_became_visible = (not previous_visible) and self.object_visible_latch
        return self.diagnostics(ratio)

    def diagnostics(self, ratio: float | None = None) -> LeftVisibilityDiagnostics:
        return LeftVisibilityDiagnostics(
            ratio=ratio,
            raw_state=self.raw_state,
            stable_state=self.stable_state,
            visible=self.object_visible_latch,
        )


# ----------------------------- Rendering -------------------------------------


def create_object_rotation(angle_y: float) -> np.ndarray:
    cx, sx = math.cos(FIXED_X_TILT), math.sin(FIXED_X_TILT)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    return ry @ rx


def transform_points(points: np.ndarray, rotation: np.ndarray, cube_size: float) -> np.ndarray:
    camera = points @ rotation.T
    camera[:, 2] += cube_size * CUBE_CAMERA_DISTANCE_FACTOR
    return camera


def project_points(camera_points: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    z = np.maximum(camera_points[:, 2], 1.0)
    scale = CUBE_FOCAL_LENGTH / (CUBE_FOCAL_LENGTH + z)
    projected = np.empty((len(camera_points), 2), dtype=np.float32)
    projected[:, 0] = center[0] + camera_points[:, 0] * scale
    projected[:, 1] = center[1] - camera_points[:, 1] * scale
    return projected


def cube_vertices(size: float) -> np.ndarray:
    h = size * 0.5
    return np.array(
        [[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h], [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]],
        dtype=np.float32,
    )


def draw_outer_cube(frame: np.ndarray, pose: ObjectPose, rotation: np.ndarray) -> None:
    vertices = cube_vertices(pose.cube_size)
    camera = transform_points(vertices, rotation, pose.cube_size)
    projected = project_points(camera, pose.anchor)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    median_z = float(np.median(camera[:, 2]))
    for start, end in sorted(edges, key=lambda e: (camera[e[0], 2] + camera[e[1], 2]) * 0.5, reverse=True):
        depth = float((camera[start, 2] + camera[end, 2]) * 0.5)
        color = CUBE_LINE_COLOR if depth <= median_z else CUBE_REAR_LINE_COLOR
        thickness = CUBE_LINE_THICKNESS if depth <= median_z else 1
        alpha = CUBE_FRONT_ALPHA if depth <= median_z else CUBE_REAR_ALPHA
        overlay = frame.copy()
        cv2.line(overlay, tuple(np.round(projected[start]).astype(int)), tuple(np.round(projected[end]).astype(int)), color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


class MeshRenderer:
    def __init__(self) -> None:
        self.light_dir = np.array([0.25, -0.55, -0.80], dtype=np.float32)
        self.light_dir /= np.linalg.norm(self.light_dir)

    def render(self, frame: np.ndarray, pose: ObjectPose, model: ObjModel | None, angle_y: float) -> dict:
        rotation = create_object_rotation(angle_y)
        draw_outer_cube(frame, pose, rotation)
        visible = 0
        if model is not None:
            visible = self.render_model(frame, pose, model, rotation)
            draw_outer_cube(frame, pose, rotation)
        return {"rotation": rotation, "visible_triangles": visible}

    def render_transition(self, frame: np.ndarray, pose: ObjectPose, old_model: ObjModel | None, new_model: ObjModel | None, angle_y: float, progress: float) -> dict:
        progress = float(np.clip(progress, 0.0, 1.0))
        rotation = create_object_rotation(angle_y)
        draw_outer_cube(frame, pose, rotation)
        visible = 0
        if old_model is not None and progress < 0.72:
            visible += self.render_model_effect(frame, pose, old_model, rotation, alpha=1.0 - progress, scale_multiplier=1.0, dissolve=progress)
        if new_model is not None and progress > 0.18:
            emerge = (progress - 0.18) / 0.82
            visible += self.render_model_effect(frame, pose, new_model, rotation, alpha=emerge, scale_multiplier=0.62 + 0.38 * emerge, dissolve=0.0)
        self.draw_smoke(frame, pose, progress)
        draw_outer_cube(frame, pose, rotation)
        return {"rotation": rotation, "visible_triangles": visible}

    def draw_smoke(self, frame: np.ndarray, pose: ObjectPose, progress: float) -> None:
        center = np.array(pose.anchor, dtype=np.float32)
        fade = math.sin(progress * math.pi)
        for i in range(SMOKE_PARTICLE_COUNT):
            angle = i * 2.399963 + progress * 4.5
            radius = pose.cube_size * (0.08 + 0.48 * progress) * ((i % 9) + 1) / 9.0
            drift = np.array([math.cos(angle), math.sin(angle * 0.83)], dtype=np.float32) * radius
            point = center + drift + np.array([0.0, -pose.cube_size * 0.10 * progress], dtype=np.float32)
            smoke_radius = int(max(2, pose.cube_size * (0.012 + 0.018 * fade) * (1.0 + (i % 4) * 0.25)))
            color = ((80 + i * 23) % 255, (140 + i * 47) % 255, (210 + i * 31) % 255)
            overlay = frame.copy()
            cv2.circle(overlay, tuple(np.round(point).astype(int)), smoke_radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.18 * fade, frame, 1.0 - 0.18 * fade, 0.0, frame)

    def render_model(self, frame: np.ndarray, pose: ObjectPose, model: ObjModel, rotation: np.ndarray) -> int:
        return self.render_model_effect(frame, pose, model, rotation, alpha=1.0, scale_multiplier=1.0, dissolve=0.0)

    def render_model_effect(
        self,
        frame: np.ndarray,
        pose: ObjectPose,
        model: ObjModel,
        rotation: np.ndarray,
        alpha: float = 1.0,
        scale_multiplier: float = 1.0,
        dissolve: float = 0.0,
    ) -> int:
        local_vertices = model.vertices * pose.cube_size
        if scale_multiplier != 1.0:
            local_vertices = local_vertices * scale_multiplier
        camera_vertices = transform_points(local_vertices, rotation, pose.cube_size)
        projected = project_points(camera_vertices, pose.anchor)

        camera_normals = model.face_normals @ rotation.T
        visible = (camera_normals[:, 2] * model.front_normal_sign) > 0
        face_camera = camera_vertices[model.faces]
        near = np.all(face_camera[:, :, 2] > 1.0, axis=1)
        face_projected = projected[model.faces]
        frame_h, frame_w = frame.shape[:2]
        bbox_visible = (
            (face_projected[:, :, 0].max(axis=1) >= 0)
            & (face_projected[:, :, 0].min(axis=1) < frame_w)
            & (face_projected[:, :, 1].max(axis=1) >= 0)
            & (face_projected[:, :, 1].min(axis=1) < frame_h)
        )
        edge_a = face_projected[:, 1] - face_projected[:, 0]
        edge_b = face_projected[:, 2] - face_projected[:, 0]
        areas = np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]) * 0.5
        mask = visible & near & bbox_visible & (areas >= MIN_PROJECTED_TRIANGLE_AREA)
        if not np.any(mask):
            return 0

        indices = np.flatnonzero(mask)
        if dissolve > 0.0:
            noise = ((indices * 1103515245 + 12345) & 0xFFFF).astype(np.float32) / 65535.0
            indices = indices[noise > dissolve * 0.92]
            if len(indices) == 0:
                return 0
        depths = face_camera[indices, :, 2].mean(axis=1)
        order = indices[np.argsort(depths)[::-1]]
        shades = np.clip(0.32 + 0.78 * np.maximum(0.0, camera_normals[order] @ -self.light_dir), 0.20, 1.0)
        colors = np.clip(model.face_colors[order].astype(np.float32) * shades[:, None], 0, 255).astype(np.uint8)
        alphas = (model.face_alphas[order].astype(np.float32) / 255.0) * float(np.clip(alpha, 0.0, 1.0))

        for idx, color_arr, alpha in zip(order, colors, alphas):
            if alpha <= 0.02:
                continue
            pts = face_projected[idx]
            if dissolve > 0.0:
                jitter_seed = float((idx * 2654435761) & 0xFFFF) / 65535.0
                jitter_angle = jitter_seed * math.tau + dissolve * 4.0
                jitter = np.array([math.cos(jitter_angle), math.sin(jitter_angle)], dtype=np.float32) * pose.cube_size * 0.22 * dissolve
                pts = pts + jitter
            color = tuple(int(c) for c in color_arr)
            points = np.round(pts).astype(np.int32)
            if alpha >= 0.98:
                cv2.fillConvexPoly(frame, points, color, cv2.LINE_AA)
            else:
                x0 = max(int(points[:, 0].min()), 0)
                y0 = max(int(points[:, 1].min()), 0)
                x1 = min(int(points[:, 0].max()) + 1, frame.shape[1])
                y1 = min(int(points[:, 1].max()) + 1, frame.shape[0])
                if x1 <= x0 or y1 <= y0:
                    continue
                roi = frame[y0:y1, x0:x1]
                local_points = points - np.array([x0, y0], dtype=np.int32)
                overlay = roi.copy()
                cv2.fillConvexPoly(overlay, local_points, color, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)
        return len(order)


class RotationController:
    def __init__(self) -> None:
        self.angle_y = 0.0
        self.last_time: float | None = None

    def update(self, now: float) -> float:
        if self.last_time is None:
            self.last_time = now
            return self.angle_y
        delta_time = min(max(now - self.last_time, 0.0), MAX_DELTA_TIME)
        self.last_time = now
        self.angle_y = (self.angle_y + Y_ROTATION_RADIANS_PER_SECOND * delta_time) % math.tau
        return self.angle_y


class ModelTransition:
    def __init__(self, duration: float = MODEL_TRANSITION_SECONDS) -> None:
        self.duration = duration
        self.old_model: ObjModel | None = None
        self.new_model: ObjModel | None = None
        self.start_time = -float("inf")

    def start(self, old_model: ObjModel | None, new_model: ObjModel | None, now: float) -> None:
        if old_model is None or new_model is None or old_model is new_model:
            self.old_model = None
            self.new_model = None
            return
        self.old_model = old_model
        self.new_model = new_model
        self.start_time = now

    def active(self, now: float) -> bool:
        return self.old_model is not None and self.new_model is not None and (now - self.start_time) < self.duration

    def progress(self, now: float) -> float:
        return float(np.clip((now - self.start_time) / max(self.duration, 1e-6), 0.0, 1.0))


# ----------------------------- Debug drawing ---------------------------------


def draw_debug(
    frame: np.ndarray,
    show_debug_info: bool,
    models: ModelManager,
    angle_y: float,
    left_detected: bool,
    right_detected: bool,
    right_fist: RightFistDiagnostics,
    left_visibility: LeftVisibilityDiagnostics,
    mapping_swapped: bool,
    fps: float,
    demo_mode: bool,
    visible_triangles: int,
) -> None:
    if show_debug_info:
        lines = [
            f"MODELS: {len(models.models)}",
            f"CURRENT: {models.index + 1}/{len(models.models)} {models.current.name if models.current else 'none'}",
            f"ANGLE: {int(math.degrees(angle_y)) % 360:03d}",
            f"LEFT DETECTED: {'YES' if left_detected else 'NO'}",
            f"RIGHT DETECTED: {'YES' if right_detected else 'NO'}",
            f"LEFT RATIO: {left_visibility.ratio:.3f}" if left_visibility.ratio is not None else "LEFT RATIO: none",
            f"LEFT VIS: {left_visibility.raw_state}/{left_visibility.stable_state} latch={left_visibility.visible}",
            f"RIGHT FIST: {right_fist.gesture}/{right_fist.stable_state}",
            f"RIGHT EXT/FOLD: {right_fist.extended_count}/{right_fist.folded_count}",
            f"ARMED: {'YES' if right_fist.armed else 'NO'}  COOLDOWN: {right_fist.cooldown_remaining:.2f}s",
            f"HAND MAPPING: {'SWAPPED' if mapping_swapped else 'NORMAL'}",
            f"TRIANGLES: {visible_triangles}",
            f"FPS: {fps:.1f}",
            "DEMO MODE" if demo_mode else "HAND MODE",
        ]
        y = 24
        for line in lines:
            cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            y += 20


# ----------------------------- App loop --------------------------------------


def normalize_camera_frame(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    return frame


def camera_frame_quality(frame: np.ndarray | None) -> tuple[bool, str, np.ndarray | None]:
    normalized = normalize_camera_frame(frame)
    if normalized is None:
        return False, "frame is empty or not a BGR-compatible image", None
    height, width = normalized.shape[:2]
    if height < 120 or width < 160:
        return False, f"unreasonable dimensions {width}x{height}", normalized

    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    nearly_black = gray <= 8
    black_ratio = float(np.mean(nearly_black))
    if black_ratio >= 0.80:
        return False, f"{black_ratio:.0%} of pixels are nearly black", normalized

    meaningful = gray > 18
    ys, xs = np.where(meaningful)
    if len(xs) == 0:
        return False, "no meaningful non-black pixels", normalized
    row_span = (int(ys.max()) - int(ys.min()) + 1) / max(height, 1)
    col_span = (int(xs.max()) - int(xs.min()) + 1) / max(width, 1)
    if row_span < 0.45:
        return False, f"meaningful pixels confined to a narrow horizontal strip ({row_span:.2f} frame height)", normalized
    if col_span < 0.45:
        return False, f"meaningful pixels confined to a narrow vertical strip ({col_span:.2f} frame width)", normalized

    varied_tiles = 0
    for tile_y in range(4):
        for tile_x in range(4):
            y0, y1 = height * tile_y // 4, height * (tile_y + 1) // 4
            x0, x1 = width * tile_x // 4, width * (tile_x + 1) // 4
            tile = gray[y0:y1, x0:x1]
            if tile.size and float(np.mean(tile > 18)) > 0.02 and float(np.std(tile)) > 2.5:
                varied_tiles += 1
    if varied_tiles < 6:
        return False, f"meaningful variation appears in too few spatial tiles ({varied_tiles}/16)", normalized
    return True, "ok", normalized


def is_valid_camera_frame(frame: np.ndarray | None) -> bool:
    valid, _reason, _normalized = camera_frame_quality(frame)
    return valid


def backend_attempts(camera_backend: str) -> list[tuple[str, int | None]]:
    key = camera_backend.lower()
    if key == "auto":
        return [("MSMF", cv2.CAP_MSMF), ("default", None)]
    if key == "msmf":
        return [("MSMF", cv2.CAP_MSMF)]
    if key == "default":
        return [("default", None)]
    if key == "dshow":
        return [("DSHOW", cv2.CAP_DSHOW)]
    raise ValueError(f"Unknown camera backend: {camera_backend}")


class CameraStream:
    def __init__(self, index: int = CAMERA_INDEX, camera_backend: str = "auto") -> None:
        self.index = index
        self.attempts = backend_attempts(camera_backend)
        self.attempt_index = -1
        self.cap: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.sequence = 0
        self.capture_timestamp = 0.0
        self.backend_name = ""
        self.last_error = ""
        self._accepted_for_backend = 0
        self._reported_rejections: set[str] = set()

    def start(self) -> None:
        if not self.switch_to_next_backend():
            raise RuntimeError(f"Could not open webcam with a valid frame. Last error: {self.last_error}")

    def switch_to_next_backend(self) -> bool:
        self.stop_current()
        for next_index in range(self.attempt_index + 1, len(self.attempts)):
            self.attempt_index = next_index
            name, backend = self.attempts[next_index]
            cap = cv2.VideoCapture(self.index) if backend is None else cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                self.last_error = f"{name} did not open"
                print(f"Camera backend rejected: {name} - {self.last_error}")
                continue

            self.cap = cap
            self.backend_name = name
            self.stop_event.clear()
            self._accepted_for_backend = 0
            self._reported_rejections = set()
            self.thread = threading.Thread(target=self._reader_loop, args=(cap, self.stop_event, name), name=f"CameraStream-{name}", daemon=True)
            self.thread.start()

            deadline = time.monotonic() + CAMERA_START_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                with self.lock:
                    accepted = self._accepted_for_backend
                    frame_ready = self.latest_frame is not None
                if accepted >= CAMERA_VALIDATION_FRAMES and frame_ready:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"Camera backend selected: {name} ({width}x{height} native, output {CAMERA_OUTPUT_SIZE[0]}x{CAMERA_OUTPUT_SIZE[1]}, {fps:.1f} FPS reported)")
                    return True
                time.sleep(0.02)

            self.last_error = f"{name} did not deliver {CAMERA_VALIDATION_FRAMES} consecutive valid frames"
            print(f"Camera backend rejected: {name} - {self.last_error}")
            self.stop_current()
        return False

    def _reader_loop(self, cap: cv2.VideoCapture, stop_event: threading.Event, backend_name: str) -> None:
        while not stop_event.is_set():
            ok, raw = cap.read()
            if not ok:
                reason = "read returned False"
                self.last_error = f"{backend_name} {reason}"
                self._accepted_for_backend = 0
                if reason not in self._reported_rejections:
                    print(f"Camera frame rejected ({backend_name}): {reason}")
                    self._reported_rejections.add(reason)
                time.sleep(0.01)
                continue
            valid, reason, frame = camera_frame_quality(raw)
            if not valid or frame is None:
                self.last_error = f"{backend_name} rejected frame: {reason}"
                self._accepted_for_backend = 0
                if reason not in self._reported_rejections:
                    print(f"Camera frame rejected ({backend_name}): {reason}")
                    self._reported_rejections.add(reason)
                continue
            if frame.shape[1] != CAMERA_OUTPUT_SIZE[0] or frame.shape[0] != CAMERA_OUTPUT_SIZE[1]:
                frame = cv2.resize(frame, CAMERA_OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.latest_frame = frame.copy()
                self.sequence += 1
                self.capture_timestamp = time.monotonic()
                self._accepted_for_backend += 1
                self.last_error = ""

    def get_latest(self) -> tuple[np.ndarray | None, int, float, str]:
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return frame, self.sequence, self.capture_timestamp, self.backend_name

    def stop_current(self) -> None:
        self.stop_event.set()
        if self.cap is not None:
            self.cap.release()
        if self.thread is not None:
            self.thread.join(timeout=0.35)
        self.thread = None
        self.cap = None
        self.stop_event = threading.Event()
        with self.lock:
            self.latest_frame = None
            self.sequence = 0
            self.capture_timestamp = 0.0
            self._accepted_for_backend = 0

    def close(self) -> None:
        self.stop_current()


def render_display_frame(
    camera_frame: np.ndarray,
    renderer: MeshRenderer,
    pose: ObjectPose | None,
    model: ObjModel | None,
    angle_y: float,
    render_object: bool,
) -> tuple[np.ndarray, int]:
    display_frame = camera_frame.copy()
    visible_triangles = 0
    if render_object and pose is not None:
        info = renderer.render(display_frame, pose, model, angle_y)
        visible_triangles = int(info["visible_triangles"])
    return display_frame, visible_triangles


def run_camera_test(camera_backend: str = "auto") -> int:
    camera = CameraStream(CAMERA_INDEX, camera_backend)
    frames: list[tuple[int, float, np.ndarray]] = []
    try:
        camera.start()
        deadline = time.monotonic() + CAMERA_START_TIMEOUT_SECONDS
        last_sequence = -1
        while time.monotonic() < deadline and len(frames) < CAMERA_VALIDATION_FRAMES:
            frame, sequence, timestamp, backend_name = camera.get_latest()
            if frame is not None and sequence != last_sequence:
                valid, reason, normalized = camera_frame_quality(frame)
                if valid and normalized is not None:
                    frames.append((sequence, timestamp, normalized.copy()))
                    last_sequence = sequence
                else:
                    print(f"Camera test rejected latest frame ({backend_name}): {reason}")
            time.sleep(0.02)

        if len(frames) < CAMERA_VALIDATION_FRAMES:
            print(f"Camera test failed: collected {len(frames)}/{CAMERA_VALIDATION_FRAMES} fresh valid frames.")
            return 1

        backend_name = camera.backend_name
        timestamps = [item[1] for item in frames]
        intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0.0
        approx_fps = 1.0 / avg_interval if avg_interval > 1e-6 else 0.0
        diffs = [
            float(np.mean(cv2.absdiff(frames[i - 1][2], frames[i][2])))
            for i in range(1, len(frames))
        ]
        changing = any(diff > 0.15 for diff in diffs)
        fresh = len({item[0] for item in frames}) == len(frames)
        output_dir = Path("debug_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"camera_test_{backend_name.lower()}.jpg"
        cv2.imwrite(str(output_path), frames[-1][2])

        print(f"Camera test selected backend: {backend_name}")
        print(f"Camera test frame shape: {frames[-1][2].shape}")
        print(f"Camera test frame dtype: {frames[-1][2].dtype}")
        print(f"Camera test fresh frames: {'yes' if fresh else 'no'} ({len(frames)} collected)")
        print(f"Camera test changing frames: {'yes' if changing else 'no'} (mean diffs: {[round(d, 3) for d in diffs[:5]]})")
        print(f"Camera test approximate interval/FPS: {avg_interval * 1000.0:.1f} ms / {approx_fps:.1f} FPS")
        print(f"Camera test saved frame: {output_path}")
        return 0
    except Exception as exc:
        print(f"Camera test failed: {exc}")
        return 1
    finally:
        camera.close()
        cv2.destroyAllWindows()


def run_app(camera_backend: str = "auto") -> None:
    models = ModelManager(Path(__file__).resolve().parent)
    camera = CameraStream(CAMERA_INDEX, camera_backend)
    camera.start()
    tracker = HandTracker()
    resolver = HandRoleResolver()
    smoother = LeftHandPoseSmoother()
    right_fist_state = RightFistStateMachine()
    left_visibility_state = LeftVisibilityStateMachine()
    rotation = RotationController()
    renderer = MeshRenderer()
    transition = ModelTransition()

    show_debug_info = DEFAULT_SHOW_DEBUG_INFO
    show_landmarks = DEFAULT_SHOW_LANDMARKS
    demo_mode = False
    fps = 0.0
    last_fps_time = time.monotonic()
    last_hand_log = ""
    frame_index = 0
    last_results = None
    last_profile_print = 0.0
    last_sequence = -1
    left_emerge_start_time = -float("inf")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            t_frame = time.perf_counter()
            t_capture = time.perf_counter()
            frame, sequence, capture_timestamp, backend_name = camera.get_latest()
            capture_ms = (time.perf_counter() - t_capture) * 1000.0
            now = time.monotonic()
            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key).lower() if key != 255 else ""
            if key_char == "q" or key == 27:
                break
            if frame is None or sequence == last_sequence:
                stale_age = now - capture_timestamp if capture_timestamp > 0.0 else float("inf")
                if stale_age >= CAMERA_STALE_SECONDS:
                    print(f"Camera backend stale: {backend_name or 'none'} ({stale_age:.2f}s without a fresh valid frame). Trying next backend.")
                    if not camera.switch_to_next_backend():
                        raise RuntimeError(f"Camera stopped producing fresh valid frames. Last error: {camera.last_error}")
                    last_sequence = -1
                continue
            last_sequence = sequence
            camera_frame = cv2.flip(frame, 1)
            dt = max(now - last_fps_time, 1e-6)
            last_fps_time = now
            fps = fps * 0.85 + (1.0 / dt) * 0.15
            angle_y = rotation.update(now)

            t_mp = time.perf_counter()
            is_new_mediapipe_result = frame_index % MEDIAPIPE_EVERY_N_FRAMES == 0
            if is_new_mediapipe_result:
                last_results = tracker.process(camera_frame)
            results = last_results
            mediapipe_ms = (time.perf_counter() - t_mp) * 1000.0
            frame_index += 1

            hands = resolver.resolve(results)
            if show_debug_info:
                hand_log = "; ".join(f"{h['mp_label']}->{h['physical_label']}" for h in hands)
                if hand_log and hand_log != last_hand_log:
                    print(f"Hand labels: {hand_log}")
                    last_hand_log = hand_log

            left = next((hand for hand in hands if hand["physical_label"] == "Left"), None)
            right = next((hand for hand in hands if hand["physical_label"] == "Right"), None)
            left_ratio = thumb_index_ratio(left["landmarks"], camera_frame.shape) if left is not None else None
            right_gesture = None
            extended_count = folded_count = 0
            if right is not None:
                right_gesture, extended_count, folded_count, _angles = classify_right_open_fist(right["landmarks"], camera_frame.shape)
            if is_new_mediapipe_result:
                right_changed, right_fist_diag = right_fist_state.update(right_gesture, now, extended_count, folded_count)
                left_visibility_diag = left_visibility_state.update(left_ratio)
                if left_visibility_state.just_became_visible:
                    left_emerge_start_time = now
            else:
                right_changed = False
                right_fist_diag = right_fist_state.diagnostics(now)
                left_visibility_diag = left_visibility_state.diagnostics(left_ratio)
            if right_changed:
                changed_pair = models.next()
                if changed_pair is not None:
                    transition.start(changed_pair[0], changed_pair[1], now)

            if key_char == "n":
                changed_pair = models.next()
                if changed_pair is not None:
                    transition.start(changed_pair[0], changed_pair[1], now)
            elif key_char == "b":
                changed_pair = models.previous()
                if changed_pair is not None:
                    transition.start(changed_pair[0], changed_pair[1], now)
            elif key_char == "h":
                resolver.toggle()
            elif key_char == "d":
                demo_mode = not demo_mode
                print(f"Demo mode: {'ON' if demo_mode else 'OFF'}")
            elif key_char == "i":
                show_debug_info = not show_debug_info
                print(f"Debug info: {'ON' if show_debug_info else 'OFF'}")
            elif key_char == "l":
                show_landmarks = not show_landmarks
                print(f"Hand landmarks: {'ON' if show_landmarks else 'OFF'}")

            if demo_mode:
                height, width = camera_frame.shape[:2]
                demo_size = min(MAX_CUBE_SIZE_PX, max(MIN_CUBE_SIZE_PX, min(width, height) * 0.42))
                pose = ObjectPose((width // 2, height // 2), demo_size, demo_size)
                render_object = left_visibility_state.object_visible_latch
            else:
                emerge_progress = 1.0
                if left_visibility_state.object_visible_latch and now - left_emerge_start_time < LEFT_EMERGE_SECONDS:
                    t = np.clip((now - left_emerge_start_time) / LEFT_EMERGE_SECONDS, 0.0, 1.0)
                    emerge_progress = float(t * t * (3.0 - 2.0 * t))
                pose = smoother.update(left["landmarks"] if left else None, camera_frame.shape, emerge_progress)
                render_object = left is not None and left_visibility_state.object_visible_latch

            render_ms = 0.0
            t_render = time.perf_counter()
            display_frame = camera_frame.copy()
            visible_triangles = 0
            if render_object and pose is not None:
                if transition.active(now):
                    info = renderer.render_transition(display_frame, pose, transition.old_model, transition.new_model, angle_y, transition.progress(now))
                else:
                    info = renderer.render(display_frame, pose, models.current, angle_y)
                visible_triangles = int(info["visible_triangles"])
            render_ms = (time.perf_counter() - t_render) * 1000.0

            if show_landmarks:
                tracker.draw_landmarks(display_frame, results)
            draw_debug(
                display_frame,
                show_debug_info,
                models,
                angle_y,
                left is not None,
                right is not None,
                right_fist_diag,
                left_visibility_diag,
                resolver.swap_handedness,
                fps,
                demo_mode,
                visible_triangles,
            )
            if show_debug_info and now - last_profile_print > 1.0:
                total_ms = (time.perf_counter() - t_frame) * 1000.0
                print(
                    "Profile ms: "
                    f"capture={capture_ms:.1f} mediapipe={mediapipe_ms:.1f} "
                    f"render={render_ms:.1f} total={total_ms:.1f}"
                )
                last_profile_print = now
            cv2.imshow(WINDOW_NAME, display_frame)
    finally:
        tracker.close()
        camera.close()
        cv2.destroyAllWindows()


# ----------------------------- Self-test -------------------------------------


def projected_bbox_width(model: ObjModel, angle: float, pose: ObjectPose) -> float:
    rotation = create_object_rotation(angle)
    camera = transform_points(model.vertices * pose.cube_size, rotation, pose.cube_size)
    projected = project_points(camera, pose.anchor)
    return float(projected[:, 0].max() - projected[:, 0].min())


def write_orientation_renders(manager: ModelManager) -> None:
    ORIENTATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer = MeshRenderer()
    pose = ObjectPose((320, 260), 190.0, 190.0)
    for model in manager.models:
        for angle in (0, 45, 90, 180, 270):
            frame = np.zeros((520, 640, 3), dtype=np.uint8)
            renderer.render(frame, pose, model, math.radians(angle))
            output = ORIENTATION_OUTPUT_DIR / f"{model.name.replace('.obj', '')}_{angle:03d}.png"
            cv2.imwrite(str(output), frame)
    print(f"Orientation renders written to: {ORIENTATION_OUTPUT_DIR.resolve()}")


def benchmark_renderer(manager: ModelManager, frames_per_model: int = 300) -> dict[str, float]:
    renderer = MeshRenderer()
    pose = ObjectPose((320, 260), 190.0, 190.0)
    results: dict[str, float] = {}
    for model in manager.models:
        frame = np.zeros((520, 640, 3), dtype=np.uint8)
        start = time.perf_counter()
        for i in range(frames_per_model):
            frame.fill(0)
            renderer.render(frame, pose, model, (i / frames_per_model) * math.tau)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / frames_per_model) * 1000.0
        results[model.name] = avg_ms
        fps = 1000.0 / max(avg_ms, 1e-6)
        print(f"Benchmark {model.name}: {avg_ms:.2f} ms/frame, renderer-only FPS {fps:.1f}")
    return results


def run_self_test() -> None:
    manager = ModelManager(Path(__file__).resolve().parent)
    selectable = [model.path.suffix.lower() for model in manager.models]
    if any(ext != ".obj" for ext in selectable):
        raise AssertionError("Non-OBJ selectable model found")
    if not manager.models:
        raise AssertionError("No valid OBJ models found")

    pose = ObjectPose((320, 260), 190.0, 190.0)
    renderer = MeshRenderer()
    rotation0 = create_object_rotation(0.0)
    rotation90 = create_object_rotation(math.radians(90))
    expected0 = np.eye(3, dtype=np.float32)
    expected90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    if not np.allclose(rotation0, expected0, atol=1e-6) or not np.allclose(rotation90, expected90, atol=1e-6):
        raise AssertionError("Object rotation must be upright Y-axis rotation only")

    class FakeLandmark:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    class FakeLandmarks:
        def __init__(self) -> None:
            self.landmark = [FakeLandmark(0.5, 0.5) for _ in range(21)]

    fake = FakeLandmarks()
    fake.landmark[4] = FakeLandmark(0.25, 0.50)
    fake.landmark[8] = FakeLandmark(0.75, 0.50)
    fake.landmark[5] = FakeLandmark(0.35, 0.72)
    fake.landmark[17] = FakeLandmark(0.65, 0.72)
    fake.landmark[0] = FakeLandmark(0.50, 0.92)
    fake.landmark[9] = FakeLandmark(0.50, 0.60)
    midpoint_smoother = LeftHandPoseSmoother(alpha=1.0)
    tiny_pose = midpoint_smoother.update(fake, (480, 640, 3), scale=0.05)
    full_pose = midpoint_smoother.update(fake, (480, 640, 3), scale=1.0)
    if tiny_pose is None or full_pose is None:
        raise AssertionError("Left midpoint pose was not produced")
    if abs(full_pose.anchor[0] - 320) > 1 or abs(full_pose.anchor[1] - 240) > 1:
        raise AssertionError(f"Left pose anchor is not thumb/index midpoint: {full_pose.anchor}")
    if not (tiny_pose.cube_size < full_pose.cube_size and abs(full_pose.cube_size - full_pose.target_size) < 1e-6):
        raise AssertionError("Left emergence pose did not grow from tiny to full target size")

    checker = np.indices((240, 320)).sum(axis=0) % 2
    camera_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    camera_frame[checker == 0] = (30, 90, 180)
    camera_frame[checker == 1] = (180, 220, 40)
    valid_camera, reason, _normalized = camera_frame_quality(camera_frame)
    if not valid_camera:
        raise AssertionError(f"Synthetic valid camera frame was rejected: {reason}")
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    valid_black, _reason, _normalized = camera_frame_quality(black_frame)
    if valid_black:
        raise AssertionError("All-black camera frame was accepted")
    strip_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    strip_noise = np.indices((36, 640)).sum(axis=0).astype(np.uint8)
    strip_frame[:36, :, 0] = strip_noise
    strip_frame[:36, :, 1] = 255 - strip_noise
    strip_frame[:36, :, 2] = strip_noise // 2
    valid_strip, _reason, _normalized = camera_frame_quality(strip_frame)
    if valid_strip:
        raise AssertionError("Narrow-strip noisy black camera frame was accepted")
    if [name for name, _backend in backend_attempts("auto")] != ["MSMF", "default"]:
        raise AssertionError("Auto camera backend must not include DSHOW")
    if [name for name, _backend in backend_attempts("dshow")] != ["DSHOW"]:
        raise AssertionError("DSHOW must be explicit only")

    hidden_frame, hidden_triangles = render_display_frame(camera_frame, renderer, pose, manager.current, 0.0, False)
    if hidden_triangles != 0 or not np.array_equal(hidden_frame, camera_frame):
        raise AssertionError("Hidden render path did not preserve the camera frame exactly")
    if hidden_frame.shape != camera_frame.shape or hidden_frame.dtype != np.uint8:
        raise AssertionError("Display frame shape/dtype changed in hidden render path")
    visible_frame, visible_triangles = render_display_frame(camera_frame, renderer, pose, manager.current, math.radians(30), True)
    if visible_frame.shape != camera_frame.shape or visible_frame.dtype != np.uint8:
        raise AssertionError("Display frame shape/dtype changed in visible render path")
    if int(visible_frame.sum()) == 0:
        raise AssertionError("Visible render returned a black frame")
    diff = np.any(visible_frame != camera_frame, axis=2)
    if visible_triangles <= 0 or not np.any(diff):
        raise AssertionError("Visible render did not alter the camera copy")
    ys, xs = np.where(diff)
    outside = np.ones(diff.shape, dtype=bool)
    outside[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = False
    if np.any(outside) and not np.array_equal(visible_frame[outside], camera_frame[outside]):
        raise AssertionError("Renderer changed pixels outside its drawn region")

    for model in manager.models:
        if not model.path.exists():
            raise AssertionError(f"{model.name} selected path does not exist: {model.path}")
        if model.path.parent.name.lower() == "models":
            root_candidate = Path(__file__).resolve().parent / "assets" / model.name
            if root_candidate.exists() and root_candidate.stat().st_mtime > model.path.stat().st_mtime:
                raise AssertionError(f"{model.name} selected stale assets/models copy instead of newer root asset")
        if not model.source_fingerprint or len(model.source_fingerprint) != 64:
            raise AssertionError(f"{model.name} missing source fingerprint")
        changed_config = ModelConfig(model.name, rotation_x_deg=1.0)
        changed_fingerprint = model_source_fingerprint(model.path, changed_config, [], [])
        if changed_fingerprint == model.source_fingerprint:
            raise AssertionError(f"{model.name} fingerprint did not change after config change")
        if len(model.vertices) == 0 or len(model.faces) == 0:
            raise AssertionError(f"{model.name} missing vertices/faces")
        if model.faces.shape[1] != 3:
            raise AssertionError(f"{model.name} has non-triangulated faces")
        bounds = model.normalized_bounds
        if not np.isfinite(bounds[0]).all() or not np.isfinite(bounds[1]).all():
            raise AssertionError(f"{model.name} has invalid bounds")
        if np.max(bounds[1] - bounds[0]) > MODEL_FIT_RATIO + 1e-4:
            raise AssertionError(f"{model.name} not normalized into fit ratio")

        widths = [projected_bbox_width(model, math.radians(a), pose) for a in (0, 45, 90, 180, 270)]
        if max(widths) - min(widths) < 4.0:
            raise AssertionError(f"{model.name} projected geometry did not change with rotation: {widths}")
        if widths[2] > max(widths[0], widths[3]) * 1.25:
            raise AssertionError(f"{model.name} 90 degree silhouette unexpectedly larger than front/back: {widths}")
        frame = np.zeros((520, 680, 3), dtype=np.uint8)
        info = renderer.render(frame, pose, model, math.radians(45))
        if info["visible_triangles"] <= 0 or int(frame.sum()) <= 0:
            raise AssertionError(f"{model.name} did not render visible geometry")

    old_rotation = 1.234
    old_index = manager.index
    manager.next()
    if len(manager.models) > 1 and manager.index == old_index:
        raise AssertionError("Next model did not change index")
    for _ in range(len(manager.models)):
        manager.next()
    if manager.index != (old_index + 1) % len(manager.models):
        raise AssertionError("Model index did not wrap correctly")
    if old_rotation != 1.234:
        raise AssertionError("Model change reset rotation")

    if DEFAULT_SHOW_DEBUG_INFO or DEFAULT_SHOW_LANDMARKS:
        raise AssertionError("Debug text and landmarks must default to hidden")

    left_visibility = LeftVisibilityStateMachine()
    states = []
    for ratio in (1.00, 0.95, 0.25, 0.20, 0.20):
        states.append(left_visibility.update(ratio).visible)
    hide_transitions = sum(1 for before, after in zip(states, states[1:]) if before and not after)
    if hide_transitions != 1 or states[-1] is not False:
        raise AssertionError(f"Left pinch should hide exactly once and stay hidden: {states}")
    states = []
    for ratio in (0.40, 0.60, 0.90, 0.95):
        states.append(left_visibility.update(ratio).visible)
    if states[:2] != [False, False] or states[-1] is not True:
        raise AssertionError(f"Left wide gesture did not re-show only after stable wide separation: {states}")

    machine = RightFistStateMachine()
    changes = 0
    now = 0.0
    for gesture in ("CLOSED", "CLOSED", "CLOSED"):
        now += 0.2
        changed, _ = machine.update(gesture, now, folded_count=4)
        changes += int(changed)
    if changes != 0:
        raise AssertionError("Closed fist should only re-arm, not change model")
    for gesture in ("OPEN", "OPEN", "OPEN"):
        now += 0.2
        changed, _ = machine.update(gesture, now, extended_count=4)
        changes += int(changed)
    if changes != 1:
        raise AssertionError(f"Stable OPEN should change once, got {changes}")
    for gesture in ("OPEN", "OPEN", "OPEN", "OPEN"):
        now += 0.2
        changed, _ = machine.update(gesture, now, extended_count=4)
        changes += int(changed)
    if changes != 1:
        raise AssertionError("Holding open repeated the model change")
    for gesture in ("CLOSED", "CLOSED", "CLOSED"):
        now += 0.2
        changed, _ = machine.update(gesture, now, folded_count=4)
        changes += int(changed)
    if changes != 1:
        raise AssertionError("Closing fist changed model instead of only re-arming")
    for gesture in ("OPEN", "OPEN", "OPEN"):
        now += 0.2
        changed, _ = machine.update(gesture, now, extended_count=4)
        changes += int(changed)
    if changes != 2:
        raise AssertionError("Second stable OPEN did not change exactly once")

    left_only_index = manager.index
    left_visibility.update(0.20)
    left_visibility.update(0.20)
    left_visibility.update(0.20)
    if manager.index != left_only_index:
        raise AssertionError("Left pinch changed model index")
    visible_before_right = left_visibility.object_visible_latch
    for gesture in ("CLOSED", "CLOSED", "CLOSED", "OPEN", "OPEN", "OPEN"):
        now += 0.2
        changed, _ = machine.update(gesture, now, extended_count=4 if gesture == "OPEN" else 0, folded_count=4 if gesture == "CLOSED" else 0)
        if changed:
            manager.next()
    if left_visibility.object_visible_latch != visible_before_right:
        raise AssertionError("Right fist changed left visibility latch")

    rotation_test = RotationController()
    a0 = rotation_test.update(10.0)
    a1 = rotation_test.update(10.5)
    manager.next()
    a2 = rotation_test.update(11.0)
    hidden_latch = left_visibility.object_visible_latch
    a3 = rotation_test.update(11.5)
    if not (a1 != a0 and a2 != a1 and a3 != a2):
        raise AssertionError("Rotation did not advance while visible/hidden/model-changing")
    if left_visibility.object_visible_latch != hidden_latch:
        raise AssertionError("Rotation update changed visibility latch")

    transition = ModelTransition()
    pair = manager.next()
    if pair is None:
        raise AssertionError("Model cycling failed to produce transition pair")
    transition.start(pair[0], pair[1], 20.0)
    if not transition.active(20.25):
        raise AssertionError("Model transition did not become active")
    transition_frame = camera_frame.copy()
    transition_info = renderer.render_transition(transition_frame, pose, pair[0], pair[1], math.radians(30), 0.5)
    if transition_info["visible_triangles"] <= 0 or np.array_equal(transition_frame, camera_frame):
        raise AssertionError("Model transition did not render visible dissolve/emerge effect")

    write_orientation_renders(manager)
    print("Self-test passed: OBJ loading, normalization, rendering, right-open switching, left visibility, transition, compositing.")


def run_benchmark() -> None:
    manager = ModelManager(Path(__file__).resolve().parent)
    benchmark_renderer(manager)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe OBJ model box")
    parser.add_argument("--self-test", action="store_true", help="run offline validation")
    parser.add_argument("--benchmark", action="store_true", help="run renderer benchmark without webcam")
    parser.add_argument("--camera-test", action="store_true", help="Test webcam capture without loading models or MediaPipe")
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "msmf", "default", "dshow"),
        default="auto",
        help="camera backend; auto tries MSMF then default, dshow is explicit only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.camera_test:
        sys.exit(run_camera_test(args.camera_backend))
    if args.self_test:
        run_self_test()
    elif args.benchmark:
        run_benchmark()
    else:
        run_app(args.camera_backend)


if __name__ == "__main__":
    main()
