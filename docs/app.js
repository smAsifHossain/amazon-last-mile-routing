"use strict";

/* ------------------------------------------------------------------ helpers */
const $ = (selector, root = document) => root.querySelector(selector);
const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, parent = null, text = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  if (text !== null) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}
function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
}
const mean = values => values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);
const score = value => value.toFixed(4);
const percent = value => Math.round(value * 100) + "%";
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const shortZone = name => (name.includes("-") ? name.slice(name.indexOf("-") + 1) : name);
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const shortDate = iso => `${+iso.slice(8, 10)} ${MONTHS[+iso.slice(5, 7) - 1]}`;

function niceScale(maxValue) {
  for (const step of [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]) {
    if (maxValue / step <= 6) {
      const top = Math.ceil(maxValue / step - 1e-9) * step;
      const ticks = [];
      for (let v = 0; v <= top + 1e-9; v += step) ticks.push(+v.toFixed(4));
      return { top, ticks, digits: step < 0.1 ? 2 : 1 };
    }
  }
  return { top: maxValue, ticks: [0, maxValue], digits: 1 };
}

/* ------------------------------------------------------------------ tooltip */
const tip = $("#tooltip");
const tipRow = (label, value) => `<div class="row"><span>${label}</span><span>${value}</span></div>`;
function showTip(event, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  let x = event.clientX + 14, y = event.clientY + 14;
  if (x + box.width > innerWidth - 8) x = event.clientX - box.width - 14;
  if (y + box.height > innerHeight - 8) y = event.clientY - box.height - 14;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
}
const hideTip = () => { tip.hidden = true; };

/* Numbers count up from zero the first time they scroll into view. */
const counted = new WeakSet();
const counter = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    counter.unobserve(entry.target);
    const { value, format } = entry.target.countTo;
    const started = performance.now();
    const tick = now => {
      const progress = Math.min(Math.max((now - started) / 1300, 0), 1);
      entry.target.textContent = format(value * (1 - Math.pow(1 - progress, 3)));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}, { threshold: 0.4 });
function countUp(node, value, format) {
  node.textContent = format(value);
  if (counted.has(node) || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  counted.add(node);
  node.countTo = { value, format };
  counter.observe(node);
}

/* -------------------------------------------------------------------- theme */
function isDark() {  // dark is the default look
  return document.documentElement.dataset.theme !== "light";
}
try {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
} catch (error) { /* storage unavailable: follow the system setting */ }
$("#theme").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (error) { /* not remembered */ }
  onThemeChange();
});

/* Zone order is an ordinal scale: one hue, light to dark. Each theme uses the
   steps that stay visible on its own surface. */
const RAMP_LIGHT = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const RAMP_DARK = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"];
const VEHICLE = "#eb6834";
function rampColor(t) {
  const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
  const x = Math.min(Math.max(t, 0), 1) * (ramp.length - 1);
  const i = Math.min(Math.floor(x), ramp.length - 2), f = x - i;
  const a = parseInt(ramp[i].slice(1), 16), b = parseInt(ramp[i + 1].slice(1), 16);
  const mix = shift => Math.round(((a >> shift) & 255) * (1 - f) + ((b >> shift) & 255) * f);
  return "#" + [16, 8, 0].map(shift => mix(shift).toString(16).padStart(2, "0")).join("");
}
function inkOn(hex) {
  const v = parseInt(hex.slice(1), 16);
  const lum = (0.2126 * ((v >> 16) & 255) + 0.7152 * ((v >> 8) & 255) + 0.0722 * (v & 255)) / 255;
  return lum > 0.6 ? "#0b0b0b" : "#ffffff";
}

/* ------------------------------------------------------------------- charts */
function barPath(x, y, w, h) {  // horizontal bar: square at the baseline, rounded data-end
  const r = Math.min(4, w, h / 2);
  return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`;
}
function columnPath(x, y, w, h) {  // column: square at the baseline, rounded top
  const r = Math.min(4, h, w / 2);
  return `M${x},${y + h}v${-(h - r)}a${r},${r} 0 0 1 ${r},${-r}h${w - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z`;
}

function chartMethods(host, rows) {
  host.textContent = "";
  const width = Math.max(host.clientWidth, 300);
  const labelW = width < 440 ? 128 : 186, valueW = 58, rowH = 42, barH = 18;
  const height = rows.length * rowH + 6;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Mean score by method" }, host);
  const scale = (width - labelW - valueW) / Math.max(...rows.map(row => row.value));
  rows.forEach((row, i) => {
    const y = i * rowH + (rowH - barH) / 2, w = Math.max(row.value * scale, 2);
    const label = svg("text", { x: 0, y: y + barH / 2 - 2, class: "t-label", "dominant-baseline": "middle" }, root, row.name);
    if (row.note) svg("tspan", { x: 0, dy: 14, class: "t-note" }, label, row.note);
    svg("path", { d: barPath(labelW, y, w, barH), class: (row.accent ? "m-accent" : "m-base") + " m-bar" }, root);
    svg("text", { x: labelW + w + 8, y: y + barH / 2 + 1, class: "t-value", "dominant-baseline": "middle" }, root, score(row.value));
    const hit = svg("rect", { x: 0, y: i * rowH, width, height: rowH, class: "m-hit" }, root);
    hit.addEventListener("mousemove", event => showTip(event, `<b>${row.name}</b>${tipRow("Average score", score(row.value))}${tipRow("Routes", row.count.toLocaleString())}<div>${row.tip}</div>`));
    hit.addEventListener("mouseleave", hideTip);
  });
  svg("line", { x1: labelW, x2: labelW, y1: 2, y2: height - 2, class: "axis" }, root);
}

function chartCities(host, cities) {
  host.textContent = "";
  const width = Math.max(host.clientWidth, 300), height = 232;
  const m = { left: 38, right: 6, top: 18, bottom: 26 };
  const plotW = width - m.left - m.right, plotH = height - m.top - m.bottom;
  const { top, ticks, digits } = niceScale(Math.max(...cities.map(c => c.shortest)));
  const y = value => m.top + plotH - (value / top) * plotH;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Mean score by city for the shortest route and the learned model" }, host);
  for (const tick of ticks) {
    svg("line", { x1: m.left, x2: width - m.right, y1: y(tick), y2: y(tick), class: tick === 0 ? "axis" : "grid" }, root);
    svg("text", { x: m.left - 6, y: y(tick) + 4, class: "t-tick", "text-anchor": "end" }, root, tick.toFixed(digits));
  }
  const band = plotW / cities.length, barW = Math.min(24, band * 0.26), gap = 2;
  cities.forEach((city, i) => {
    const centre = m.left + band * (i + 0.5);
    svg("path", { d: columnPath(centre - barW - gap / 2, y(city.shortest), barW, y(0) - y(city.shortest)), class: "m-base m-col" }, root);
    svg("path", { d: columnPath(centre + gap / 2, y(city.learned), barW, y(0) - y(city.learned)), class: "m-accent m-col" }, root);
    svg("text", { x: centre + gap / 2 + barW / 2, y: y(city.learned) - 5, class: "t-value", "text-anchor": "middle" }, root, city.learned.toFixed(3));
    svg("text", { x: centre, y: height - 8, class: "t-label", "text-anchor": "middle" }, root, band < 86 ? city.short : city.city);
    const hit = svg("rect", { x: m.left + band * i, y: m.top, width: band, height: plotH + m.bottom, class: "m-hit" }, root);
    hit.addEventListener("mousemove", event => showTip(event, `<b>${city.city}</b>${tipRow("Shortest route", score(city.shortest))}${tipRow("This model", score(city.learned))}${tipRow("Test routes", city.count)}`));
    hit.addEventListener("mouseleave", hideTip);
  });
}

function chartScatter(host, routes, detailIds) {
  host.textContent = "";
  const width = Math.max(host.clientWidth, 300), height = Math.max(280, Math.min(400, width * 0.42));
  const m = { left: 44, right: 16, top: 26, bottom: 40 };
  const plotW = width - m.left - m.right, plotH = height - m.top - m.bottom;
  const { top, ticks, digits } = niceScale(Math.max(...routes.map(r => Math.max(r.shortest, r.learned))));
  const x = value => m.left + (value / top) * plotW, y = value => m.top + plotH - (value / top) * plotH;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Learned model score against shortest route score for every held-out route" }, host);
  for (const tick of ticks) {
    svg("line", { x1: m.left, x2: width - m.right, y1: y(tick), y2: y(tick), class: tick === 0 ? "axis" : "grid" }, root);
    svg("line", { x1: x(tick), x2: x(tick), y1: m.top, y2: m.top + plotH, class: tick === 0 ? "axis" : "grid" }, root);
    svg("text", { x: m.left - 6, y: y(tick) + 4, class: "t-tick", "text-anchor": "end" }, root, tick.toFixed(digits));
    svg("text", { x: x(tick), y: m.top + plotH + 16, class: "t-tick", "text-anchor": "middle" }, root, tick.toFixed(digits));
  }
  svg("line", { x1: x(0), y1: y(0), x2: x(top), y2: y(top), class: "axis" }, root);
  svg("text", { x: x(top) - 6, y: y(top) + 14, class: "t-note", "text-anchor": "end" }, root, "equally close");
  svg("text", { x: m.left, y: 12, class: "t-label" }, root, "This model's score");
  svg("text", { x: width - m.right, y: height - 4, class: "t-label", "text-anchor": "end" }, root, "Shortest route's score");

  const order = routes.map((_, i) => i).sort((a, b) => detailIds.has(routes[a].id) - detailIds.has(routes[b].id));
  for (const i of order) {
    const route = routes[i], hasDetail = detailIds.has(route.id);
    const dot = svg("circle", { cx: x(route.shortest), cy: y(route.learned), r: 4, class: "m-dot" + (hasDetail ? " has-detail" : ""), "data-i": i }, root);
    dot.style.transitionDelay = (i % 40) * 15 + "ms";
  }
  root.addEventListener("mousemove", event => {
    const i = event.target.dataset && event.target.dataset.i;
    if (i === undefined) { hideTip(); return; }
    const route = routes[i];
    showTip(event, `<b>${route.city} · ${route.station}</b>${tipRow("Shortest route", score(route.shortest))}${tipRow("This model", score(route.learned))}${tipRow("Stops / areas", route.stops + " / " + route.zones)}` +
      (detailIds.has(route.id) ? "<div>Click to open on the map</div>" : ""));
  });
  root.addEventListener("mouseleave", hideTip);
  root.addEventListener("click", event => {
    const i = event.target.dataset && event.target.dataset.i;
    if (i !== undefined && detailIds.has(routes[i].id)) location.hash = "#/explorer/" + routes[i].id;
  });
}

function table(host, header, rows) {
  host.textContent = "";
  const node = el("table");
  const head = el("tr");
  header.forEach(name => head.appendChild(el("th", "", name)));
  node.appendChild(head);
  rows.forEach(row => {
    const line = el("tr");
    row.forEach(cell => line.appendChild(el("td", "", String(cell))));
    node.appendChild(line);
  });
  host.appendChild(node);
}

/* ----------------------------------------------------------------- overview */
const state = { summary: null, details: [], detailIds: new Set(), stats: null };

function computeStats() {
  const routes = state.summary.routes;
  const by = name => mean(routes.map(r => r[name]));
  const cityNames = [...new Set(routes.map(r => r.city))].sort();
  state.stats = {
    count: routes.length,
    shortest: by("shortest"), fixed: by("fixed"), learned: by("learned"), known: by("known"),
    improved: mean(routes.map(r => (r.learned < r.shortest ? 1 : 0))),
    firstRight: mean(routes.map(r => (r.first_zone_right ? 1 : 0))),
    movesRight: by("moves_right"),
    cities: cityNames.map(city => {
      const group = routes.filter(r => r.city === city);
      return {
        city, short: city.split(" ").map(word => word[0]).join("").length > 1 ? city.split(" ").map(word => word[0]).join("") : city.slice(0, 3),
        count: group.length, shortest: mean(group.map(r => r.shortest)), learned: mean(group.map(r => r.learned)),
      };
    }),
  };
}

function renderOverview() {
  const s = state.stats, fitted = state.summary.fitted_on, official = state.summary.split === "eval";
  const what = official ? "official evaluation routes" : "held-out training routes";
  const gain = 1 - s.learned / s.shortest;
  countUp($("#hero-value"), gain, percent);
  countUp($("#p-gain"), gain, percent);
  countUp($("#p-routes"), s.count, value => Math.round(value).toLocaleString());
  $("#hero-compare").textContent = `Average score: this model ${score(s.learned)}, shortest route ${score(s.shortest)}.`;
  $("#p-routes-label").textContent = official ? "official test routes it had never seen" : "unseen routes used for testing";
  countUp($("#p-moves"), s.movesRight, percent);
  $("#f-train").textContent = fitted.toLocaleString();
  $("#r-kicker").textContent = `Results on ${s.count.toLocaleString()} ${what}`;
  $("#r-note").textContent = official
    ? "Scored with the challenge's official metric on its official evaluation routes. For reference, the winning entry of the 2021 challenge scored 0.0248 and the runner-up 0.0353. The challenge is closed, so this is a self-reported score and not a leaderboard entry: it was not run on the organisers' machines or under their time limits."
    : "These figures are from training routes held out of fitting, not from the challenge's official evaluation set.";

  const tiles = $("#tiles");
  tiles.textContent = "";
  [
    ["Test routes", s.count.toLocaleString(), `the model learned from ${fitted.toLocaleString()} other routes`],
    ["Routes improved", percent(s.improved), "closer to the driver than the shortest route"],
    ["Starts in the right area", percent(s.firstRight), "routes where the model begins where the driver began"],
    ["Moves between areas right", percent(s.movesRight), "of the drivers' moves from one area to the next"],
  ].forEach(([label, value, hint]) => {
    const tile = el("div", "card tile");
    tile.append(el("div", "name", label), el("div", "value", value), el("div", "hint", hint));
    tiles.appendChild(tile);
  });

  const methods = [
    { name: "Shortest route", note: "what a map app suggests", value: s.shortest, tip: "The route with the least driving, ignoring how drivers behave." },
    { name: "Simple area rules", note: "hand-written, no learning", value: s.fixed, tip: "Shortest route with a fixed rule: finish one area before starting the next." },
    { name: "This model", note: "learned from drivers", value: s.learned, accent: true, tip: "The order of areas is predicted from past routes, then every stop is ordered inside it." },
    { name: "Perfect area order", note: "best this approach could do", value: s.known, tip: "What the model would score if it always guessed the order of areas correctly." },
  ].map(row => ({ ...row, count: s.count }));
  chartMethods($("#chart-methods"), methods);
  table($("#table-methods"), ["Method", "Average score"], methods.map(row => [row.name, score(row.value)]));
  chartCities($("#chart-cities"), s.cities);
  table($("#table-cities"), ["City", "Routes", "Shortest route", "This model"], s.cities.map(c => [c.city, c.count, score(c.shortest), score(c.learned)]));
  $("#scatter-title").textContent = `On ${percent(s.improved)} of routes, this model is closer to the driver`;
  chartScatter($("#chart-scatter"), state.summary.routes, state.detailIds);

  requestAnimationFrame(() => requestAnimationFrame(() => {
    for (const chart of document.querySelectorAll(".chart")) chart.classList.add("in");
  }));
}

/* ----------------------------------------------------------------- explorer */
const ex = {
  city: "All", sort: "gain", route: null, method: "learned", t: 0, playing: false, speed: 3, autoPlayed: false,
  left: null, right: null, hoverZone: null, zoneColor: [], lastFrame: 0, syncing: false,
};
const STOPS_PER_SECOND = 4;

function makePanel(id, whereId) {
  const map = L.map(id, { zoomSnap: 0.25, preferCanvas: true });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  return { map, layer: L.layerGroup().addTo(map), where: $(whereId), markers: [], pos: [], shown: [], lastI: -1 };
}
function ensureMaps() {
  if (ex.left) return;
  ex.left = makePanel("map-left", "#where-left");
  ex.right = makePanel("map-right", "#where-right");
  const link = (a, b) => a.map.on("move", () => {
    if (ex.syncing) return;
    ex.syncing = true;
    b.map.setView(a.map.getCenter(), a.map.getZoom(), { animate: false });
    ex.syncing = false;
  });
  link(ex.left, ex.right);
  link(ex.right, ex.left);
}

function drawPanel(panel, route, name) {
  panel.layer.clearLayers();
  const sequence = route.sequences[name], n = sequence.length;
  const at = stop => route.coords[stop];
  const ink = cssVar("--ink"), surface = cssVar("--surface"), faint = cssVar("--muted");
  panel.sequence = sequence;
  panel.name = name;
  panel.pos = new Array(route.coords.length).fill(0);
  sequence.forEach((stop, i) => { panel.pos[stop] = i; });
  panel.minutes = [0];
  route.legs[name].forEach(leg => panel.minutes.push(panel.minutes[panel.minutes.length - 1] + leg / 60));

  const stationLeg = { color: faint, weight: 1.5, dashArray: "4 6", interactive: false };
  L.polyline([at(sequence[0]), at(sequence[1])], stationLeg).addTo(panel.layer);
  L.polyline([at(sequence[n - 1]), at(sequence[0])], stationLeg).addTo(panel.layer);
  L.polyline(sequence.slice(1).map(at), { color: ink, weight: 1.5, opacity: 0.22, interactive: false }).addTo(panel.layer);
  panel.done = L.polyline([], { color: ink, weight: 2.5, opacity: 0.9, interactive: false }).addTo(panel.layer);

  panel.markers = new Array(route.coords.length).fill(null);
  panel.shown = new Array(route.coords.length).fill(-1);
  for (let i = 1; i < n; i++) {
    const stop = sequence[i], zone = route.stop_zone[stop];
    const marker = L.circleMarker(at(stop), { radius: 5, color: surface, weight: 1.5, fillColor: ex.zoneColor[zone], fillOpacity: 1 });
    marker.on("mouseover", event => showTip(event.originalEvent,
      `<b>Area ${route.zone_names[zone]}</b>` +
      tipRow("On this map", `stop ${i} of ${n - 1}`) +
      tipRow("For the driver", `stop ${ex.left.pos[stop]}`) +
      tipRow("Packages", route.packages[stop])));
    marker.on("mouseout", hideTip);
    marker.addTo(panel.layer);
    panel.markers[stop] = marker;
  }
  L.circleMarker(at(sequence[0]), { radius: 6, color: surface, weight: 2, fillColor: ink, fillOpacity: 1 })
    .on("mouseover", event => showTip(event.originalEvent, `<b>Depot ${route.station}</b><div>Every route starts and ends here.</div>`))
    .on("mouseout", hideTip).addTo(panel.layer);
  panel.vehicle = L.circleMarker(at(sequence[0]), { radius: 8, color: surface, weight: 3, fillColor: VEHICLE, fillOpacity: 1, interactive: false }).addTo(panel.layer);
  panel.lastI = -1;
}

function restyle(panel, route, i) {
  for (let stop = 0; stop < panel.markers.length; stop++) {
    const marker = panel.markers[stop];
    if (!marker) continue;
    const dim = ex.hoverZone !== null && route.stop_zone[stop] !== ex.hoverZone;
    const opacity = dim ? 0.1 : panel.pos[stop] <= i ? 1 : 0.28;
    if (panel.shown[stop] !== opacity) {
      marker.setStyle({ fillOpacity: opacity, opacity: dim ? 0.15 : 1 });
      panel.shown[stop] = opacity;
    }
  }
}

function updatePanel(panel, route, t, force) {
  const sequence = panel.sequence, n = sequence.length;
  const i = Math.min(Math.floor(t + 1e-9), n - 1), f = Math.min(t - i, 1);
  const a = route.coords[sequence[i]], b = route.coords[sequence[Math.min(i + 1, n - 1)]];
  const here = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
  panel.vehicle.setLatLng(here);
  const path = [];
  for (let j = 1; j <= i; j++) path.push(route.coords[sequence[j]]);
  if (i >= 1 && f > 0) path.push(here);
  panel.done.setLatLngs(path);
  if (i !== panel.lastI || force) {
    restyle(panel, route, i);
    panel.lastI = i;
    panel.where.textContent = i === 0
      ? "At the depot"
      : `Area ${shortZone(route.zone_names[route.stop_zone[sequence[i]]])} · ${Math.round(panel.minutes[i])} min driving`;
  }
  return i === 0 ? null : route.stop_zone[sequence[i]];
}

function markNow(host, zone) {
  for (const chip of host.querySelectorAll(".zchip")) chip.classList.toggle("now", zone !== null && +chip.dataset.zone === zone);
}

function setTime(t, force = false) {
  const route = ex.route;
  if (!route) return;
  const last = route.sequences.driver.length - 1;
  ex.t = Math.min(Math.max(t, 0), last);
  $("#scrub").value = ex.t;
  const zoneLeft = updatePanel(ex.left, route, ex.t, force);
  const zoneRight = updatePanel(ex.right, route, ex.t, force);
  markNow($("#z-driver"), zoneLeft);
  markNow($("#z-method"), zoneRight);
  $("#readout").textContent = `Stop ${Math.min(Math.floor(ex.t + 1e-9), last)} of ${last}`;
}

function setPlaying(playing) {
  ex.playing = playing;
  $("#play-icon").setAttribute("d", playing ? "M3 1.5h3v11H3zM8 1.5h3v11H8z" : "M3 1.5v11l9-5.5z");
  $("#play").setAttribute("aria-label", playing ? "Pause" : "Play route");
  $("#play").classList.toggle("idle", !playing);
  if (playing) {
    const last = ex.route.sequences.driver.length - 1;
    if (ex.t >= last - 1e-6) setTime(0, true);
    ex.lastFrame = performance.now();
    requestAnimationFrame(frame);
  }
}
function frame(now) {
  if (!ex.playing) return;
  const last = ex.route.sequences.driver.length - 1;
  const next = ex.t + Math.min((now - ex.lastFrame) / 1000, 0.1) * STOPS_PER_SECOND * ex.speed;
  ex.lastFrame = now;
  setTime(next);
  if (next >= last) setPlaying(false);
  else requestAnimationFrame(frame);
}

function renderZones(route) {
  const driverOrder = route.zone_orders.driver, methodOrder = route.zone_orders[ex.method];
  const driverRank = new Array(route.zone_names.length).fill(0);
  driverOrder.forEach((zone, i) => { driverRank[zone] = i; });
  const driverMoves = new Set(driverOrder.slice(1).map((zone, i) => driverOrder[i] + ">" + zone));
  const sizes = new Array(route.zone_names.length).fill(0);
  route.stop_zone.forEach(zone => { if (zone >= 0) sizes[zone]++; });

  const fill = (host, order, markBreaks) => {
    host.textContent = "";
    order.forEach((zone, i) => {
      if (markBreaks && i > 0 && !driverMoves.has(order[i - 1] + ">" + zone)) {
        const mark = el("i", "zbreak");
        mark.title = "The driver did not go from the previous area to this one";
        host.appendChild(mark);
      }
      const chip = el("span", "zchip", shortZone(route.zone_names[zone]));
      chip.dataset.zone = zone;
      chip.style.background = ex.zoneColor[zone];
      chip.style.color = inkOn(ex.zoneColor[zone]);
      chip.addEventListener("mouseenter", event => {
        ex.hoverZone = zone;
        for (const other of document.querySelectorAll(".zchip")) other.classList.toggle("dim", +other.dataset.zone !== zone);
        restyle(ex.left, route, ex.left.lastI);
        restyle(ex.right, route, ex.right.lastI);
        showTip(event, `<b>Area ${route.zone_names[zone]}</b>${tipRow("Stops", sizes[zone])}${tipRow("Driver reached it", `${driverRank[zone] + 1} of ${driverOrder.length}`)}${tipRow("In this row", `${order.indexOf(zone) + 1} of ${order.length}`)}`);
      });
      chip.addEventListener("mouseleave", () => {
        ex.hoverZone = null;
        for (const other of document.querySelectorAll(".zchip")) other.classList.remove("dim");
        restyle(ex.left, route, ex.left.lastI);
        restyle(ex.right, route, ex.right.lastI);
        hideTip();
      });
      host.appendChild(chip);
    });
  };
  fill($("#z-driver"), driverOrder, false);
  fill($("#z-method"), methodOrder, true);
  $("#z-who").textContent = ex.method === "learned" ? "This model" : "Shortest route";
}

const ICON_RIGHT = '<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><circle cx="9" cy="9" r="9" fill="currentColor"/><path d="M5 9.4 7.8 12 13 6.5" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ICON_WRONG = '<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><circle cx="9" cy="9" r="9" fill="currentColor"/><path d="M6 6l6 6M12 6l-6 6" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>';

function renderStats(route) {
  const minutes = name => Math.round(route.legs[name].reduce((a, b) => a + b, 0) / 60);
  const host = $("#r-stats");
  host.textContent = "";
  const add = (label, valueNode, hint) => {
    const card = el("div", "card stat" + (host.children.length === 0 ? " lead" : ""));
    const value = el("div", "value");
    if (typeof valueNode === "string") value.textContent = valueNode; else value.appendChild(valueNode);
    card.append(el("div", "name", label), value, el("div", "hint", hint));
    host.appendChild(card);
  };
  add("This model's score", score(route.learned), "difference from the driver; 0 = identical");
  add("Shortest route's score", score(route.shortest), `${(route.shortest / Math.max(route.learned, 1e-4)).toFixed(1)}× further from the driver`);
  add("Area moves matched", percent(route.moves_right), "of the driver's moves between areas");
  const verdict = el("span", "verdict " + (route.first_zone_right ? "good" : "bad"));
  verdict.innerHTML = (route.first_zone_right ? ICON_RIGHT : ICON_WRONG) + `<span>${route.first_zone_right ? "Right" : "Wrong"}</span>`;
  add("Starting area", verdict, "does the model start where the driver did");
  add("Driving time", `${minutes("learned")} min`, `driver ${minutes("driver")} · shortest ${minutes("shortest")} min`);
}

function selectRoute(id, fit = true) {
  const firstListed = $("#list .item");
  const route = state.details.find(r => r.id === id)
    || state.details.find(r => firstListed && r.id === firstListed.dataset.id)
    || state.details[0];
  if (!route) return;
  setPlaying(false);
  ex.route = route;
  ex.hoverZone = null;
  const order = route.zone_orders.driver;
  ex.zoneColor = new Array(route.zone_names.length).fill("#898781");
  order.forEach((zone, i) => { ex.zoneColor[zone] = rampColor(order.length > 1 ? i / (order.length - 1) : 0); });

  $("#r-title").textContent = `${route.city} · depot ${route.station}`;
  $("#r-sub").textContent = `${route.stops} stops in ${route.zones} areas · ${route.date}${route.quality ? " · " + route.quality.toLowerCase() + "-quality driver sequence" : ""}`;
  renderStats(route);
  for (const item of document.querySelectorAll(".item")) item.classList.toggle("on", item.dataset.id === route.id);
  const active = $(".item.on");
  if (active) active.scrollIntoView({ block: "nearest" });

  ensureMaps();
  drawPanel(ex.left, route, "driver");
  drawPanel(ex.right, route, ex.method);
  renderZones(route);
  const last = route.sequences.driver.length - 1;
  $("#scrub").max = last;
  if (fit) {
    ex.left.map.invalidateSize();
    ex.right.map.invalidateSize();
    ex.left.map.fitBounds(route.coords.slice(1), { padding: [28, 28], animate: false });
  }
  setTime(last, true);
  if (!ex.autoPlayed && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    ex.autoPlayed = true;
    setTimeout(() => { if (ex.route === route && !$("#view-explorer").hidden) setPlaying(true); }, 700);
  }
  if (location.hash !== "#/explorer/" + route.id) history.replaceState(null, "", "#/explorer/" + route.id);
}

function renderList() {
  const sorters = {
    gain: (a, b) => (b.shortest - b.learned) - (a.shortest - a.learned),
    best: (a, b) => a.learned - b.learned,
    worst: (a, b) => b.learned - a.learned,
    stops: (a, b) => b.stops - a.stops,
  };
  const shown = state.details.filter(r => ex.city === "All" || r.city === ex.city).sort(sorters[ex.sort]);
  const host = $("#list");
  host.textContent = "";
  const widest = Math.max(...state.details.map(r => r.shortest), 0.01);
  shown.forEach(route => {
    const item = el("button", "item" + (ex.route && ex.route.id === route.id ? " on" : ""));
    item.type = "button";
    item.dataset.id = route.id;
    const num = el("span", "num");
    num.innerHTML = `${route.shortest.toFixed(3)} → <b>${route.learned.toFixed(3)}</b>`;
    const mini = el("span", "mini");
    const base = el("i"), accent = el("i", "accent");
    base.style.width = (route.shortest / widest) * 100 + "%";
    accent.style.width = (route.learned / widest) * 100 + "%";
    mini.append(base, accent);
    item.append(el("span", "name", `${route.city} · ${route.station}`), num,
      el("span", "meta", `${shortDate(route.date)} · ${route.stops} stops · ${route.zones} areas`), el("span", "meta", "shortest → this model"), mini);
    item.addEventListener("click", () => selectRoute(route.id));
    host.appendChild(item);
  });
  $("#count").textContent = `${shown.length} of ${state.details.length} routes shown`;
}

function setupExplorer() {
  const chips = $("#city-chips");
  ["All", ...new Set(state.details.map(r => r.city))].sort((a, b) => (a === "All" ? -1 : b === "All" ? 1 : a.localeCompare(b))).forEach(city => {
    const chip = el("button", "chip-btn" + (city === "All" ? " on" : ""), city);
    chip.type = "button";
    chip.addEventListener("click", () => {
      ex.city = city;
      for (const other of chips.children) other.classList.toggle("on", other === chip);
      renderList();
    });
    chips.appendChild(chip);
  });
  $("#sort").addEventListener("change", event => { ex.sort = event.target.value; renderList(); });
  $("#play").addEventListener("click", () => setPlaying(!ex.playing));
  $("#scrub").addEventListener("input", event => { setPlaying(false); setTime(+event.target.value); });
  $("#speed").addEventListener("click", event => {
    const speed = event.target.dataset.speed;
    if (!speed) return;
    ex.speed = +speed;
    for (const button of $("#speed").children) button.classList.toggle("on", button === event.target);
  });
  $("#method").addEventListener("click", event => {
    const method = event.target.dataset.method;
    if (!method || method === ex.method) return;
    ex.method = method;
    for (const button of $("#method").children) button.classList.toggle("on", button === event.target);
    drawPanel(ex.right, ex.route, method);
    renderZones(ex.route);
    setTime(ex.t, true);
  });
  document.addEventListener("keydown", event => {
    if (event.code !== "Space" || $("#view-explorer").hidden) return;
    if (["INPUT", "SELECT", "BUTTON", "A"].includes(document.activeElement.tagName)) return;
    event.preventDefault();
    setPlaying(!ex.playing);
  });
  $("#ramp").style.background = `linear-gradient(90deg, ${(isDark() ? RAMP_DARK : RAMP_LIGHT).join(",")})`;
  renderList();
}

/* ------------------------------------------------------------------ routing */
function showView() {
  const [, name = "home", arg] = location.hash.split("/");
  const view = ["home", "results", "explorer"].includes(name) ? name : "home";
  for (const section of document.querySelectorAll(".view")) section.hidden = section.id !== "view-" + view;
  for (const link of document.querySelectorAll("[data-nav]")) link.classList.toggle("on", link.dataset.nav === view);
  hideTip();
  if (view !== "explorer") { setPlaying(false); $("#view-" + view).scrollTop = 0; window.scrollTo(0, 0); }
  if (!state.summary) return;
  if (view === "explorer") {
    const wanted = arg || (ex.route && ex.route.id);
    if (!ex.route || (arg && arg !== ex.route.id) || !ex.left) selectRoute(wanted);
    else { ex.left.map.invalidateSize(); ex.right.map.invalidateSize(); }
  } else {
    renderOverview();
  }
}

function onThemeChange() {
  if (!state.summary) return;
  $("#ramp").style.background = `linear-gradient(90deg, ${(isDark() ? RAMP_DARK : RAMP_LIGHT).join(",")})`;
  if (ex.left) {
    if (ex.route) {
      const t = ex.t;
      selectRoute(ex.route.id, false);
      setTime(t, true);
    }
  }
}

let resizeTimer = 0;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state.summary && !$("#view-results").hidden) renderOverview(); }, 150);
});
addEventListener("hashchange", showView);

Promise.all([
  fetch("data/summary.json").then(response => response.json()),
  fetch("data/routes.json").then(response => response.json()),
]).then(([summary, details]) => {
  state.summary = summary;
  state.details = details;
  state.detailIds = new Set(details.map(route => route.id));
  computeStats();
  setupExplorer();
  showView();
}).catch(error => {
  $("#view-home .page").prepend(el("p", "loading", "Could not load the data files: " + error));
});

/* Hero animation: a route drawn area by area, looping. */
function heroVisual() {
  const host = $("#hero-visual");
  let seed = 7;
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const centres = [[120, 290], [150, 130], [290, 80], [410, 170], [350, 310]];
  const root = svg("svg", { viewBox: "0 0 520 390", role: "img", "aria-label": "Animated delivery route visiting five areas in turn" }, host);
  const gradient = svg("linearGradient", { id: "hv-grad", x1: 0, y1: 1, x2: 1, y2: 0 }, svg("defs", {}, root));
  svg("stop", { offset: 0, "stop-color": "#3987e5" }, gradient);
  svg("stop", { offset: 1, "stop-color": "#22d3ee" }, gradient);
  let d = "M40,360";
  for (const [cx, cy] of centres) {
    svg("circle", { cx, cy, r: 56, class: "hv-zone" }, root);
    const stops = Array.from({ length: 7 }, () => [cx + (random() - 0.5) * 80, cy + (random() - 0.5) * 80])
      .sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
    for (const [x, y] of stops) {
      svg("circle", { cx: x, cy: y, r: 3.5, class: "hv-stop" }, root);
      d += ` L${x.toFixed(1)},${y.toFixed(1)}`;
    }
  }
  d += " L40,360";
  svg("path", { d, class: "hv-ghost" }, root);
  const path = svg("path", { d, id: "hv-route", class: "hv-path", pathLength: 1, "stroke-dasharray": 1, "stroke-dashoffset": 1 }, root);
  svg("animate", { attributeName: "stroke-dashoffset", values: "1;0;0", keyTimes: "0;0.88;1", dur: "10s", repeatCount: "indefinite" }, path);
  svg("rect", { x: 33, y: 353, width: 14, height: 14, rx: 3, class: "hv-station" }, root);
  svg("text", { x: 54, y: 376, class: "hv-label" }, root, "Depot");
  centres.forEach(([cx, cy], i) => {
    const badge = svg("g", { class: "hv-badge" }, root);
    svg("circle", { cx, cy: cy - 62, r: 11 }, badge);
    svg("text", { x: cx, y: cy - 62 }, badge, String(i + 1));
  });
  const van = svg("circle", { r: 6, class: "hv-van" }, root);
  const motion = svg("animateMotion", { dur: "10s", repeatCount: "indefinite", keyPoints: "0;1;1", keyTimes: "0;0.88;1", calcMode: "linear" }, van);
  svg("mpath", { href: "#hv-route" }, motion);
}
heroVisual();

/* Sections fade up as they scroll into view. */
const revealer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (entry.isIntersecting) { entry.target.classList.add("in"); revealer.unobserve(entry.target); }
  }
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach(node => revealer.observe(node));
$("#guide-close").addEventListener("click", () => { $("#guide").hidden = true; });
showView();
