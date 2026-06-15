#!/usr/bin/env python3
"""Export GenMed comparison meshes to web-ready GLB for the project page.

For each chosen high-Dice case we export four aligned GLBs:
  gt.glb      ground-truth shape (origin)
  prompt.glb  the partial observation actually given to the model, shown as the
              observed fragment (solid) inside a faint ghost of the full shape
  cond.glb    input-conditioning baseline output
  ours.glb    GenMed (guided) output
Dice (vs GT, occupancy = SDF < 0) is computed for cond and ours and written into
manifest.json so the page can display it.

Run with the project conda env:
    /home/weidongguo/miniconda3/envs/controlseg/bin/python3 \
        project-page/genmed-project-page/tools/export_compare_glb.py
"""
import json
import shutil

import numpy as np
import open3d as o3d
import torch
import trimesh
from pathlib import Path
from skimage import measure

ROOT = Path("/mnt/nas1/disk01/weidongguo/yuheliu/msn_lyh")
OUT = ROOT / "project-page/genmed-project-page/static/models"

PROMPT_DIRS = {
    "oneplane":   ("output_vis/COND1__use_oneplane__cp20",   "output_vis/GUIDE1__use_oneplane__cp20"),
    "triplane":   ("output_vis/COND1__use_triplane__cp20",   "output_vis/GUIDE1__use_triplane__cp20"),
    "multiplane": ("output_vis/COND1__use_multiplane__cp20",  "output_vis/GUIDE1__use_multiplane__cp20"),
    "broken":     ("output_vis_orig/COND1__use_broken__cp20", "output_vis_orig/GUIDE1__use_broken__cp20"),
}

# (id, prompt_type, organ, sample, label, prompt_label) — high-Dice, Ours > COND, varied prompts
CASES = [
    ("kidney_oneplane",     "oneplane",   "kidney_right",     "042966_kidneyright",            "Kidney (right)",    "One-plane cross-section"),
    ("bladder_broken",      "broken",     "urinary_bladder",  "s1151_urinary_bladder.nii.g_1", "Urinary bladder",   "Broken / missing region"),
    ("gallbladder_triplane","triplane",   "gallbladder",      "012499_gallbladder",            "Gallbladder",       "Tri-plane cross-section"),
    ("myocardium_broken",   "broken",     "heart_myocardium", "s0513_heart_myocardium.nii.g_1","Heart myocardium",  "Broken / missing region"),
    ("eyeball_multiplane",  "multiplane", "eyeballright",     "062131_eyeballright",           "Eyeball (right)",   "Multi-plane slices"),
    ("femur_broken",        "broken",     "femur_right",      "s0624_femur_right.nii.g_1",     "Femur (right)",     "Broken / missing region"),
]

COL = {  # base RGB
    "gt":     (0.40, 0.64, 0.87),
    "cond":   (0.64, 0.70, 0.74),
    "ours":   (0.27, 0.66, 0.54),
    "ghost":  (0.45, 0.66, 0.86),
    "frag":   (0.92, 0.68, 0.24),
}
TARGET_FACES = 9000


def _np(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    while a.ndim > 3:
        a = a[0]
    return a.astype(np.float32)


def dice(pred, gt):
    a, b = (_np(pred) < 0), (_np(gt) < 0)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)


def _largest(m):
    if len(m.triangles) == 0:
        return m
    idx, counts, _ = m.cluster_connected_triangles()
    idx, counts = np.asarray(idx), np.asarray(counts)
    if counts.size:
        keep = int(np.argmax(counts))
        m.remove_triangles_by_index(np.where(idx != keep)[0].tolist())
        m.remove_unreferenced_vertices()
    return m


def field_to_vf(field, level, smooth=12, keep_largest=True):
    """Marching cubes on a scalar field -> (verts, faces) in the shared frame."""
    lo, hi = float(field.min()), float(field.max())
    if not (lo < level < hi):
        return None
    try:
        v, f, _, _ = measure.marching_cubes(field, level=level, step_size=1)
    except (ValueError, RuntimeError):
        return None
    if len(v) == 0 or len(f) == 0:
        return None
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(v)
    m.triangles = o3d.utility.Vector3iVector(f)
    if keep_largest:
        m = _largest(m)
    if smooth:
        m = m.filter_smooth_taubin(number_of_iterations=smooth)
    if len(m.triangles) > TARGET_FACES:
        m = m.simplify_quadric_decimation(TARGET_FACES)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    if len(m.triangles) == 0:
        return None
    verts = (np.asarray(m.vertices) - 32.0) / 64.0
    return verts, np.asarray(m.triangles)


def _material(rgb, alpha=1.0):
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=[rgb[0], rgb[1], rgb[2], alpha],
        metallicFactor=0.0, roughnessFactor=0.6,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
        doubleSided=True,
    )


def _mesh(vf, rgb, alpha=1.0):
    v, f = vf
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)
    trimesh.repair.fix_normals(m)
    m.visual = trimesh.visual.TextureVisuals(material=_material(rgb, alpha))
    return m


def export_simple(path, vf, rgb):
    _mesh(vf, rgb).export(path)


def export_prompt(path, gt_field, mask_field):
    """Prompt viz: the OBSERVED part (solid gold) + the UNOBSERVED part of the
    shape (faint ghost). Reads as 'what the model is given' vs 'what it must infer'."""
    occ_gt = gt_field < 0
    # mask is an SDF for the broken prompt (has negatives) but 0/1 occupancy for planes
    if float(mask_field.min()) < -1e-3:
        obs_occ = mask_field < 0.0
    else:
        obs_occ = mask_field > 0.5
    obs_field = np.where(occ_gt & obs_occ, -1.0, 1.0).astype(np.float32)
    mis_field = np.where(occ_gt & ~obs_occ, -1.0, 1.0).astype(np.float32)
    obs_vf = field_to_vf(obs_field, 0.0, smooth=4, keep_largest=False)
    mis_vf = field_to_vf(mis_field, 0.0, smooth=4, keep_largest=False)
    geoms = []
    if mis_vf is not None:
        geoms.append(_mesh(mis_vf, COL["ghost"], alpha=0.22))   # unobserved -> faint
    if obs_vf is not None:
        geoms.append(_mesh(obs_vf, COL["frag"]))                 # observed -> solid gold
    if not geoms:
        # fully observed: fall back to a single solid shape
        gt_vf = field_to_vf(gt_field, 0.0)
        if gt_vf is None:
            return False
        geoms = [_mesh(gt_vf, COL["frag"])]
    trimesh.Scene(geoms).export(path)
    return True


def main():
    if OUT.exists():
        for d in OUT.iterdir():
            if d.is_dir():
                shutil.rmtree(d)
    manifest = []
    for cid, prompt, organ, sample, label, plabel in CASES:
        cdir, gdir = (ROOT / d for d in PROMPT_DIRS[prompt])
        cs, gs = cdir / organ / sample, gdir / organ / sample
        if not (cs / "generated.pt").exists() or not (gs / "generated.pt").exists():
            print(f"[skip] {cid}: missing tensors"); continue

        gt = _np(torch.load(cs / "original.pt", map_location="cpu"))
        mask = _np(torch.load(cs / "mask.pt", map_location="cpu"))
        cond = _np(torch.load(cs / "generated.pt", map_location="cpu"))
        ours = _np(torch.load(gs / "generated.pt", map_location="cpu"))

        gt_vf, cond_vf, ours_vf = (field_to_vf(x, 0.0) for x in (gt, cond, ours))
        if gt_vf is None or ours_vf is None or cond_vf is None:
            print(f"[skip] {cid}: empty mesh"); continue

        d = OUT / cid; d.mkdir(parents=True, exist_ok=True)
        export_simple(d / "gt.glb", gt_vf, COL["gt"])
        export_simple(d / "cond.glb", cond_vf, COL["cond"])
        export_simple(d / "ours.glb", ours_vf, COL["ours"])
        has_prompt = export_prompt(d / "prompt.glb", gt, mask)

        files = {"gt": f"{cid}/gt.glb", "cond": f"{cid}/cond.glb", "ours": f"{cid}/ours.glb"}
        if has_prompt:
            files["prompt"] = f"{cid}/prompt.glb"
        entry = {
            "id": cid, "label": label, "prompt": prompt, "promptLabel": plabel,
            "diceOurs": round(dice(ours, gt), 3), "diceCond": round(dice(cond, gt), 3),
            "files": files,
        }
        manifest.append(entry)
        print(f"  {cid:22s} Dice ours={entry['diceOurs']:.3f} cond={entry['diceCond']:.3f} prompt={'y' if has_prompt else 'n'}")

    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {len(manifest)} cases -> {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
