# GenMed — Project Page

Project website for **GenMed: A Pairwise Generative Reformulation of Medical Diagnostic Tasks**.

GenMed reformulates discriminative medical tasks as a **generative model of the joint
distribution _P(X, Y)_**, turning inference into a **test-time output-optimization** problem.
A single diffusion model — trained once, with no task-specific conditioning encoder — handles
3D image segmentation (standard, cross-modality, few-shot, degraded) and 3D shape completion
(single-/multi-/tri-plane, partial, noisy, and zero-shot), all by guiding the sampling process
to match the observed input.

## Live page

Published via GitHub Pages: **https://schmidtkk.github.io/genmed-project-page/**

## Structure

```
index.html              # the page
static/css/style.css    # self-contained styles (no CSS framework dependency)
static/js/main.js        # image lightbox + copy-BibTeX + carousel
static/js/compare.js     # interactive multi-column 3D mesh comparison (model-viewer)
static/images/           # paper figures + rendered gallery panels
static/models/           # web-ready GLB meshes (GT / input / COND / Ours) per case
tools/export_compare_glb.py  # regenerate the GLBs from the .pt SDF volumes
```

The **3D comparison** section uses [`<model-viewer>`](https://modelviewer.dev/) (loaded from a
CDN) to show four camera-synced columns of real meshes (Ground truth / Input prompt / Input
conditioning / GenMed) with Dice, CD and UHD per method. The prompt column colour-codes the
partial observation: observed (gold), missing/broken-away (blue), redundant/extra (purple).
Dice is on the SDF occupancy; CD/UHD use the paper's metric (`utils/metrics.py`, ×100). To
regenerate / add cases, edit `CASES` in `tools/export_compare_glb.py`, run it with the project
conda env (needs a GPU for CD/UHD), then update the `SETS` array in `static/js/compare.js`.

## Run locally

It is a static site — just open `index.html`, or serve the folder:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Editing notes

- **Links** for Paper / arXiv / Code / Dataset are intentionally placeholders (`soon`) — the paper
  PDF is **not** published here. Update the buttons in `index.html` (search for `btn--soon`) once
  the URLs are live.
- The **Gallery** carousel is driven by `static/images/gallery_*.png`. Add/replace slides by editing
  the `#gallery` section in `index.html`.
- Author affiliations and the venue badge are at the top of `index.html` — verify before release.
- Result numbers are taken from the paper's Tables I, II, III and VI.

## Attribution

Design inspired by the [Nerfies](https://github.com/nerfies/nerfies.github.io) project-page
template, used under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/). All figures
and results are from the GenMed paper.
