/* GenMed — interactive multi-column 3D mesh comparison (model-viewer) */
(function () {
  "use strict";

  var BASE = "static/models/";

  // High-Dice cases (mirrors static/models/manifest.json). Dice vs GT (occ = SDF<0).
  var SETS = [
    { id: "bladder_broken",       label: "Urinary bladder", tag: "Broken",      ours: 0.984, cond: 0.920 },
    { id: "eyeball_multiplane",   label: "Eyeball (right)", tag: "Multi-plane", ours: 0.984, cond: 0.938 },
    { id: "femur_broken",         label: "Femur (right)",   tag: "Broken",      ours: 0.969, cond: 0.911 },
    { id: "myocardium_broken",    label: "Myocardium",      tag: "Broken",      ours: 0.961, cond: 0.858 },
    { id: "gallbladder_triplane", label: "Gallbladder",     tag: "Tri-plane",   ours: 0.934, cond: 0.719 },
    { id: "kidney_oneplane",      label: "Kidney (right)",  tag: "One-plane",   ours: 0.864, cond: 0.554 }
  ];
  SETS.forEach(function (s) {
    s.files = { gt: 1, prompt: 1, cond: 1, ours: 1 };
  });

  // columns left->right: target, the given input, baseline, ours
  var COLS = [
    { key: "gt",     name: "Ground truth",      dot: "#6aa9e0" },
    { key: "prompt", name: "Input prompt",       dot: "#e8ae3d" },
    { key: "cond",   name: "Input conditioning", dot: "#9fb0b6", badge: "cond" },
    { key: "ours",   name: "GenMed (Ours)",      dot: "#4ea089", badge: "ours", isOurs: true }
  ];

  var grid = document.getElementById("mvGrid");
  var picker = document.getElementById("setPicker");
  if (!grid || !picker) return;

  var viewers = {}, emptyEls = {}, badgeEls = {};
  var syncing = false, userPosed = false;

  COLS.forEach(function (c) {
    var col = document.createElement("div");
    col.className = "mv-col" + (c.isOurs ? " is-ours" : "");

    var head = document.createElement("div");
    head.className = "mv-col__head";
    head.innerHTML = '<span class="mv-dot" style="background:' + c.dot + '"></span><span>' + c.name + "</span>";
    if (c.badge) {
      var badge = document.createElement("span");
      badge.className = "mv-dice";
      head.appendChild(badge);
      badgeEls[c.key] = badge;
    }
    col.appendChild(head);

    var mv = document.createElement("model-viewer");
    mv.setAttribute("camera-controls", "");
    mv.setAttribute("auto-rotate", "");
    mv.setAttribute("auto-rotate-delay", "0");
    mv.setAttribute("rotation-per-second", "18deg");
    mv.setAttribute("interaction-prompt", "none");
    mv.setAttribute("shadow-intensity", "0.55");
    mv.setAttribute("shadow-softness", "1");
    mv.setAttribute("exposure", "1.05");
    mv.setAttribute("environment-image", "neutral");
    mv.setAttribute("camera-orbit", "35deg 72deg auto");
    mv.setAttribute("camera-target", "0m 0m 0m");
    mv.setAttribute("disable-pan", "");
    mv.style.setProperty("--progress-bar-color", "#0e7490");

    var empty = document.createElement("div");
    empty.className = "mv-empty";
    empty.textContent = "not available for this case";
    empty.style.display = "none";

    col.appendChild(mv);
    col.appendChild(empty);
    grid.appendChild(col);

    viewers[c.key] = mv;
    emptyEls[c.key] = empty;

    mv.addEventListener("camera-change", function (e) {
      if (syncing || !e.detail || e.detail.source !== "user-interaction") return;
      if (!userPosed) { userPosed = true; stopAutoRotate(); }
      propagate(c.key);
    });
  });

  function stopAutoRotate() {
    COLS.forEach(function (c) { viewers[c.key].removeAttribute("auto-rotate"); });
  }

  function propagate(srcKey) {
    var src = viewers[srcKey];
    var o = src.getCameraOrbit();
    var fov = src.getFieldOfView();
    var orbitStr = o.theta + "rad " + o.phi + "rad " + o.radius + "m";
    syncing = true;
    COLS.forEach(function (c) {
      if (c.key === srcKey) return;
      var v = viewers[c.key];
      v.cameraOrbit = orbitStr;
      v.fieldOfView = fov + "deg";
      if (v.jumpCameraToGoal) v.jumpCameraToGoal();
    });
    requestAnimationFrame(function () { requestAnimationFrame(function () { syncing = false; }); });
  }

  function loadSet(set) {
    COLS.forEach(function (c) {
      var mv = viewers[c.key];
      if (set.files[c.key]) {
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
    if (badgeEls.cond) badgeEls.cond.textContent = "Dice " + set.cond.toFixed(3);
    if (badgeEls.ours) badgeEls.ours.textContent = "Dice " + set.ours.toFixed(3);
  }

  SETS.forEach(function (set, idx) {
    var b = document.createElement("button");
    b.innerHTML = set.label + "<small>" + set.tag + "</small>";
    b.addEventListener("click", function () {
      picker.querySelectorAll("button").forEach(function (x) { x.classList.remove("active"); });
      b.classList.add("active");
      loadSet(set);
    });
    if (idx === 0) b.classList.add("active");
    picker.appendChild(b);
  });

  loadSet(SETS[0]);
})();
