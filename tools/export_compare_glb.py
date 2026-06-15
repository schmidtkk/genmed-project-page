#!/usr/bin/env python3
"""Export GenMed comparison meshes (GT / Input / COND / Ours) to web-ready GLB.

For each (prompt_type, organ) we locate a sample that exists under both the COND
and GUIDE result directories, marching-cubes the SDF volumes, smooth + decimate,
normalise into a shared frame so the columns align, and write one GLB per column
plus a manifest.json that the project page reads.

Run with the project conda env:
    /home/weidongguo/miniconda3/envs/controlseg/bin/python3 \
        project-page/genmed-project-page/tools/export_compare_glb.py
"""
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from skimage import measure

ROOT = Path("/mnt/nas1/disk01/weidongguo/yuheliu/msn_lyh")
OUT = ROOT / "project-page/genmed-project-page/static/models"

# prompt_type -> (COND dir, GUIDE dir)
PROMPT_DIRS = {
    "oneplane":   ("output_vis/COND1__use_oneplane__cp20",   "output_vis/GUIDE1__use_oneplane__cp20"),
    "triplane":   ("output_vis/COND1__use_triplane__cp20",   "output_vis/GUIDE1__use_triplane__cp20"),
    "multiplane": ("output_vis/COND1__use_multiplane__cp20",  "output_vis/GUIDE1__use_multiplane__cp20"),
    "broken":     ("output_vis_orig/COND1__use_broken__cp20", "output_vis_orig/GUIDE1__use_broken__cp20"),
}

# (set id, prompt_type, organ dir name, pretty label, prompt label)
WISHLIST = [
    ("bladder_oneplane",   "oneplane",   "urinary_bladder",  "Urinary bladder", "One-plane cross-section"),
    ("kidney_broken",      "broken",     "kidney_right",     "Kidney (right)",  "Broken / missing region"),
    ("myocardium_triplane","triplane",   "heart_myocardium", "Heart myocardium","Tri-plane cross-section"),
    ("femur_multiplane",   "multiplane", "femur_right",      "Femur (right)",   "Multi-plane slices"),
    ("sacrum_multiplane",  "multiplane", "sacrum",           "Sacrum",          "Multi-plane slices"),
    ("atrium_oneplane",    "oneplane",   "heart_atrium_left","Left atrium",     "One-plane cross-section"),
]

# RGB base colours (0-1) per column
COLORS = {
    "gt":    (0.42, 0.66, 0.88),   # blue   — ground truth
    "input": (0.78, 0.80, 0.84),   # grey   — partial observation
    "cond":  (0.62, 0.70, 0.74),   # slate  — conditioning baseline
    "ours":  (0.31, 0.66, 0.55),   # green  — GenMed (ours)
}

TARGET_FACES = 9000


def _to_np(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.asarray(t, dtype=np.float32)
    while t.ndim > 3:
        t = t[0]
    return t


def _largest_cluster(mesh: o3d.geometry.TriangleMesh):
    if len(mesh.triangles) == 0:
        return mesh
    idx, counts, _ = mesh.cluster_connected_triangles()
    idx = np.asarray(idx)
    counts = np.asarray(counts)
    if counts.size == 0:
        return mesh
    keep = int(np.argmax(counts))
    remove = np.where(idx != keep)[0]
    mesh.remove_triangles_by_index(remove.tolist())
    mesh.remove_unreferenced_vertices()
    return mesh


def volume_to_mesh(vol, level, smooth_iters=12):
    vol = _to_np(vol)
    lo, hi = float(vol.min()), float(vol.max())
    if not (lo < level < hi):
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=level, step_size=1)
    except (ValueError, RuntimeError):
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(faces)
    m = _largest_cluster(m)
    if smooth_iters:
        m = m.filter_smooth_taubin(number_of_iterations=smooth_iters)
    if len(m.triangles) > TARGET_FACES:
        m = m.simplify_quadric_decimation(TARGET_FACES)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    if len(m.triangles) == 0:
        return None

    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)
    # shared frame: SDF volumes live in [0,64]^3 -> center on 32, scale by 1/64
    v = (v - 32.0) / 64.0
    return v, f


def write_glb(path, v, f, rgb):
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    trimesh.repair.fix_normals(mesh)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.65,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return mesh.faces.shape[0]


def find_sample(cond_dir: Path, guide_dir: Path, organ: str):
    co = cond_dir / organ
    gu = guide_dir / organ
    if not co.is_dir() or not gu.is_dir():
        return None
    cond_samples = {p.name for p in co.iterdir() if (p / "generated.pt").exists()}
    for p in sorted(gu.iterdir()):
        if p.name in cond_samples and (p / "generated.pt").exists():
            return co / p.name, gu / p.name
    return None


def main():
    manifest = []
    for set_id, prompt, organ, label, prompt_label in WISHLIST:
        cond_root, guide_root = (ROOT / d for d in PROMPT_DIRS[prompt])
        found = find_sample(cond_root, guide_root, organ)
        if not found:
            print(f"[skip] {set_id}: no matched sample for {organ} ({prompt})")
            continue
        cond_s, guide_s = found

        gt = torch.load(cond_s / "original.pt", map_location="cpu")
        mask = torch.load(cond_s / "mask.pt", map_location="cpu")
        cond = torch.load(cond_s / "generated.pt", map_location="cpu")
        ours = torch.load(guide_s / "generated.pt", map_location="cpu")

        cols = {
            "gt":    volume_to_mesh(gt,   level=0.0),
            "input": volume_to_mesh(mask, level=0.5, smooth_iters=4),
            "cond":  volume_to_mesh(cond, level=0.0),
            "ours":  volume_to_mesh(ours, level=0.0),
        }
        if cols["gt"] is None or cols["ours"] is None:
            print(f"[skip] {set_id}: GT/Ours mesh empty")
            continue

        entry = {"id": set_id, "label": label, "prompt": prompt,
                 "promptLabel": prompt_label, "files": {}}
        for col, res in cols.items():
            if res is None:
                continue
            v, f = res
            rel = f"{set_id}/{col}.glb"
            nfaces = write_glb(OUT / rel, v, f, COLORS[col])
            entry["files"][col] = rel
            print(f"  {set_id}/{col}.glb  faces={nfaces}")
        manifest.append(entry)

    (OUT).mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {len(manifest)} comparison sets -> {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
