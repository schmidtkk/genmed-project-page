/* GenMed — interactive multi-column 3D mesh comparison (model-viewer) */
(function () {
  "use strict";

  var BASE = "static/models/";

  // Comparison sets (mirrors static/models/manifest.json)
  var SETS = [
    { id: "bladder_oneplane",    label: "Urinary bladder", prompt: "One-plane",   files: { gt: 1, input: 1, cond: 1, ours: 1 } },
    { id: "kidney_broken",       label: "Kidney (right)",  prompt: "Broken",      files: { gt: 1, input: 0, cond: 1, ours: 1 } },
    { id: "myocardium_triplane", label: "Myocardium",      prompt: "Tri-plane",   files: { gt: 1, input: 1, cond: 1, ours: 1 } },
    { id: "femur_multiplane",    label: "Femur (right)",   prompt: "Multi-plane", files: { gt: 1, input: 1, cond: 1, ours: 1 } },
    { id: "sacrum_multiplane",   label: "Sacrum",          prompt: "Multi-plane", files: { gt: 1, input: 1, cond: 1, ours: 1 } },
    { id: "atrium_oneplane",     label: "Left atrium",     prompt: "One-plane",   files: { gt: 1, input: 1, cond: 1, ours: 1 } }
  ];

  // Column metadata: key -> {name, dot color, highlight}
  var COLS = [
    { key: "gt",    name: "Ground truth",       dot: "#6aa9e0" },
    { key: "input", name: "Input prompt",        dot: "#aab2bd" },
    { key: "cond",  name: "Input conditioning",  dot: "#9fb0b6" },
    { key: "ours",  name: "GenMed (Ours)",       dot: "#4ea089", ours: true }
  ];

  var grid = document.getElementById("mvGrid");
  var picker = document.getElementById("setPicker");
  if (!grid || !picker) return;

  var viewers = {};   // key -> model-viewer element
  var emptyEls = {};  // key -> placeholder element
  var syncing = false;
  var userPosed = false;

  // ---- build the 4 columns once ----
  COLS.forEach(function (c) {
    var col = document.createElement("div");
    col.className = "mv-col" + (c.ours ? " is-ours" : "");

    var head = document.createElement("div");
    head.className = "mv-col__head";
    head.innerHTML = '<span class="mv-dot" style="background:' + c.dot + '"></span>' + c.name;
    col.appendChild(head);

    var mv = document.createElement("model-viewer");
    mv.setAttribute("camera-controls", "");
    mv.setAttribute("auto-rotate", "");
    mv.setAttribute("auto-rotate-delay", "0");
    mv.setAttribute("rotation-per-second", "18deg");
    mv.setAttribute("interaction-prompt", "none");
    mv.setAttribute("shadow-intensity", "0.6");
    mv.setAttribute("shadow-softness", "1");
    mv.setAttribute("exposure", "1.05");
    mv.setAttribute("environment-image", "neutral");
    mv.setAttribute("camera-orbit", "35deg 72deg auto");
    mv.setAttribute("camera-target", "0m 0m 0m");
    mv.setAttribute("min-camera-orbit", "auto auto auto");
    mv.setAttribute("max-camera-orbit", "auto auto auto");
    mv.setAttribute("disable-pan", "");
    mv.style.setProperty("--progress-bar-color", "#0e7490");

    var empty = document.createElement("div");
    empty.className = "mv-empty";
    empty.textContent = "no partial input for this case";
    empty.style.display = "none";

    col.appendChild(mv);
    col.appendChild(empty);
    grid.appendChild(col);

    viewers[c.key] = mv;
    emptyEls[c.key] = empty;

    // camera sync — only react to genuine user drags
    mv.addEventListener("camera-change", function (e) {
      if (syncing) return;
      if (!e.detail || e.detail.source !== "user-interaction") return;
      if (!userPosed) { userPosed = true; stopAutoRotate(); }
      propagate(c.key);
    });
  });

  function stopAutoRotate() {
    COLS.forEach(function (c) { viewers[c.key].removeAttribute("auto-rotate"); });
  }

  function propagate(srcKey) {
    var src = viewers[srcKey];
    var orbit = src.getCameraOrbit();      // {theta, phi, radius} radians/m
    var fov = src.getFieldOfView();        // degrees
    var orbitStr = orbit.theta + "rad " + orbit.phi + "rad " + orbit.radius + "m";
    syncing = true;
    COLS.forEach(function (c) {
      if (c.key === srcKey) return;
      var v = viewers[c.key];
      v.cameraOrbit = orbitStr;
      v.fieldOfView = fov + "deg";
      if (v.jumpCameraToGoal) v.jumpCameraToGoal();
    });
    // release the guard after the change settles
    requestAnimationFrame(function () { requestAnimationFrame(function () { syncing = false; }); });
  }

  // ---- load a comparison set ----
  function loadSet(set) {
    COLS.forEach(function (c) {
      var mv = viewers[c.key];
      var has = !!set.files[c.key];
      if (has) {
        mv.style.display = "block";
        emptyEls[c.key].style.display = "none";
        mv.src = BASE + set.id + "/" + c.key + ".glb";
        if (!userPosed) mv.setAttribute("auto-rotate", "");
      } else {
        mv.removeAttribute("src");
        mv.style.display = "none";
        emptyEls[c.key].style.display = "flex";
      }
    });
  }

  // ---- set picker ----
  SETS.forEach(function (set, idx) {
    var b = document.createElement("button");
    b.innerHTML = set.label + "<small>" + set.prompt + "</small>";
    b.addEventListener("click", function () {
      picker.querySelectorAll("button").forEach(function (x) { x.classList.remove("active"); });
      b.classList.add("active");
      loadSet(set);
    });
    if (idx === 0) b.classList.add("active");
    picker.appendChild(b);
  });

  // initial load
  loadSet(SETS[0]);
})();
