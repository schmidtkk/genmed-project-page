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
from scipy.ndimage import gaussian_filter, zoom
from skimage import measure

ROOT = Path("/mnt/nas1/disk01/weidongguo/yuheliu/msn_lyh")
OUT = ROOT / "project-page/genmed-project-page/static/models"

PROMPT_DIRS = {
    "oneplane":   ("output_vis/COND1__use_oneplane__cp20",   "output_vis/GUIDE1__use_oneplane__cp20"),
    "triplane":   ("output_vis/COND1__use_triplane__cp20",   "output_vis/GUIDE1__use_triplane__cp20"),
    "multiplane": ("output_vis/COND1__use_multiplane__cp20",  "output_vis/GUIDE1__use_multiplane__cp20"),
    "broken":     ("output_vis_orig/COND1__use_broken__cp20", "output_vis_orig/GUIDE1__use_broken__cp20"),
}

# (id, prompt_type, organ, sample, label, prompt_label) — Ours > COND, varied prompts.
# Broken cases chosen to have a VISIBLE missing chunk and/or redundant region.
CASES = [
    ("pancreas_broken",     "broken",     "pancreas",          "s0236_pancreas.nii.g_1",         "Pancreas",          "Broken — missing + redundant"),
    ("pulmonary_broken",    "broken",     "pulmonary_artery",  "s0080_pulmonary_artery.nii.g_1", "Pulmonary artery",  "Broken — missing + redundant"),
    ("adrenal_broken",      "broken",     "adrenal_gland_right","s0369_adrenal_gland_right.nii.g_1","Adrenal gland",   "Broken / missing region"),
    ("gallbladder_triplane","triplane",   "gallbladder",       "012499_gallbladder",             "Gallbladder",       "Tri-plane cross-section"),
    ("eyeball_multiplane",  "multiplane", "eyeballright",      "062131_eyeballright",            "Eyeball (right)",   "Multi-plane slices"),
    ("kidney_oneplane",     "oneplane",   "kidney_right",      "042966_kidneyright",             "Kidney (right)",    "One-plane cross-section"),
]

COL = {  # base RGB
    "gt":      (0.40, 0.64, 0.87),
    "cond":    (0.64, 0.70, 0.74),
    "ours":    (0.27, 0.66, 0.54),
    "match":   (0.93, 0.69, 0.24),   # observed & correct  -> gold
    "extra":   (0.70, 0.35, 0.80),   # redundant / spurious -> purple
    "missing": (0.36, 0.60, 0.88),   # broken-away / unobserved -> faint blue
}
TARGET_FACES = 11000
UPSAMPLE = 2        # trilinear-upsample the SDF before marching cubes for a smoother surface


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


def field_to_vf(field, level, smooth=10, keep_largest=True, upsample=UPSAMPLE, sigma=2.0):
    """Marching cubes on a scalar field -> (verts, faces) in the shared frame.
    The field is trilinearly upsampled and Gaussian-smoothed before MC so the
    isosurface is free of voxel terracing; coords are normalised by the
    (upsampled) resolution so all meshes stay aligned."""
    if upsample and upsample != 1:
        field = zoom(field, upsample, order=1)
        sigma = sigma * upsample
    if sigma:
        field = gaussian_filter(field, sigma=sigma)
    res = field.shape[0]
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
    m.compute_vertex_normals()
    if len(m.triangles) == 0:
        return None
    verts = (np.asarray(m.vertices) - res / 2.0) / res
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


def _occ_mesh(occ, color, alpha=1.0, thin=False):
    """Marching-cubes a boolean occupancy volume into a coloured trimesh, or None.
    `thin=True` (plane cross-section slices) uses a small fixed Gaussian so the
    thin sheets are preserved; solid regions get strong smoothing like the SDFs."""
    if occ.sum() < 12:
        return None
    field = np.where(occ, -1.0, 1.0).astype(np.float32)
    sigma = 0.3 if thin else 1.5
    vf = field_to_vf(field, 0.0, smooth=12, keep_largest=False, sigma=sigma)
    return None if vf is None else _mesh(vf, color, alpha)


def _detect_planes(occ):
    """Find axis-aligned cutting planes as (axis, index): a plane perpendicular to
    axis a shows up as a sharp peak in the per-slice coverage along a."""
    planes = []
    for a in range(3):
        cov = occ.sum(axis=tuple(i for i in range(3) if i != a)).astype(float)
        if cov.max() == 0:
            continue
        thr = max(4.0 * float(cov.mean()), 0.3 * float(cov.max()))
        for i in range(len(cov)):
            lo = max(0, i - 1); hi = min(len(cov), i + 2)
            if cov[i] >= thr and cov[i] == cov[lo:hi].max():
                planes.append((a, i))
    return planes


def _plane_quads(planes, bbox, color, alpha=0.5, margin=3):
    """Flat translucent quads (one per cutting plane), sized to the shape bbox."""
    lo = np.maximum(bbox[0] - margin, 0)
    hi = np.minimum(bbox[1] + margin, 63)
    V, F = [], []
    for a, i in planes:
        c0, c1 = [x for x in range(3) if x != a]
        base = len(V)
        for u in (lo[c0], hi[c0]):
            for w in (lo[c1], hi[c1]):
                p = [0.0, 0.0, 0.0]; p[a] = i; p[c0] = u; p[c1] = w
                V.append(p)
        F += [[base, base + 1, base + 3], [base, base + 3, base + 2]]
    if not V:
        return None
    verts = (np.asarray(V, float) - 32.0) / 64.0
    m = trimesh.Trimesh(vertices=verts, faces=np.asarray(F), process=False)
    m.visual = trimesh.visual.TextureVisuals(material=_material(color, alpha))
    return m


def export_prompt(path, gt_field, mask_field):
    """Visualise the prompt the model is given vs the true shape.
    Broken (mask is an SDF): a 3-way solid diff
        gold = observed & correct, purple = redundant/extra, blue = missing.
    Plane prompts (mask is 0/1 occupancy): the observed cutting planes drawn as
    translucent gold quads through a faint blue ghost of the full shape, so one-,
    tri- and multi-plane are equally legible."""
    occ_gt = gt_field < 0
    is_broken = float(mask_field.min()) < -1e-3

    if is_broken:
        occ_prompt = mask_field < 0.0
        match = _occ_mesh(occ_gt & occ_prompt, COL["match"])
        extra = _occ_mesh(occ_prompt & ~occ_gt, COL["extra"])
        missing = _occ_mesh(occ_gt & ~occ_prompt, COL["missing"], alpha=0.55)
        geoms = [g for g in (missing, match, extra) if g is not None]
        flags = {"match": match is not None, "extra": extra is not None, "missing": missing is not None}
    else:
        occ_prompt = mask_field > 0.5
        flags = {"planes": 0, "ghost": False, "extra": False}
        geoms = []
        ghost_vf = field_to_vf(gt_field, 0.0, sigma=2.0)
        if ghost_vf is not None:
            geoms.append(_mesh(ghost_vf, COL["missing"], alpha=0.20))
            flags["ghost"] = True
        idx = np.argwhere(occ_gt)
        bbox = (idx.min(0), idx.max(0)) if len(idx) else (np.zeros(3), np.full(3, 63))
        planes = _detect_planes(occ_prompt)
        quads = _plane_quads(planes, bbox, COL["match"], alpha=0.55)
        if quads is not None:
            geoms.append(quads)
        flags["planes"] = len(planes)
        # spurious regions outside GT (rare for planes) still shown in purple
        extra = _occ_mesh(occ_prompt & ~occ_gt, COL["extra"])
        if extra is not None:
            geoms.append(extra); flags["extra"] = True

    if not geoms:
        gt_vf = field_to_vf(gt_field, 0.0)
        if gt_vf is None:
            return flags
        geoms = [_mesh(gt_vf, COL["match"])]
    trimesh.Scene(geoms).export(path)
    return flags


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.metrics import compute_metrics_cond  # CD (pytorch3d) + UHD, paper convention

    def cd_uhd(pred, gt):
        m = compute_metrics_cond([torch.from_numpy(pred).float()],
                                 [torch.from_numpy(gt).float()])
        return round(m["cd"] * 100, 2), round(m["uhd"] * 100, 1)  # x100 to match paper scale

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
        pflags = export_prompt(d / "prompt.glb", gt, mask)
        has_prompt = any(pflags.values())

        cd_o, uhd_o = cd_uhd(ours, gt)
        cd_c, uhd_c = cd_uhd(cond, gt)

        files = {"gt": f"{cid}/gt.glb", "cond": f"{cid}/cond.glb", "ours": f"{cid}/ours.glb"}
        if has_prompt:
            files["prompt"] = f"{cid}/prompt.glb"
        entry = {
            "id": cid, "label": label, "prompt": prompt, "promptLabel": plabel,
            "promptParts": pflags,
            "diceOurs": round(dice(ours, gt), 3), "diceCond": round(dice(cond, gt), 3),
            "cdOurs": cd_o, "cdCond": cd_c, "uhdOurs": uhd_o, "uhdCond": uhd_c,
            "files": files,
        }
        manifest.append(entry)
        print(f"  {cid:22s} Dice {entry['diceCond']:.3f}->{entry['diceOurs']:.3f}  "
              f"CD {cd_c}->{cd_o}  UHD {uhd_c}->{uhd_o}  parts={[k for k,v in pflags.items() if v]}")

    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {len(manifest)} cases -> {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
