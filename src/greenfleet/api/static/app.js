/* Q Fleet. Draws what /api/* returns; computes nothing. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);

  // One hue (ocean blue), light to dark. Fossil fuels get the darkest shades, clean carriers the lightest.
  const P = { 50: "#f1f6fd", 100: "#e0ebf9", 200: "#c0d6f2", 300: "#93b8e7", 400: "#5e93d8", 500: "#3a74c4", 600: "#2a5ca8", 700: "#224a88", 800: "#1d3d6d", 900: "#172f52" };
  const FUEL_NAME = { hfo: "Heavy fuel oil", mgo: "Diesel", lng: "LNG", methanol: "Methanol", ammonia: "Ammonia", hydrogen: "Hydrogen", electricity: "Electricity" };
  const FUEL_COLOR = { hfo: "#1c2635", mgo: "#6f7c90", lng: P[800], methanol: P[600], ammonia: P[400], hydrogen: P[300], electricity: P[200] };
  const LIGHT_FUEL = new Set(["hydrogen", "electricity"]);
  const chipStyle = (f) => `background:${FUEL_COLOR[f]};color:${LIGHT_FUEL.has(f) ? P[900] : "#fff"}`;
  const PATHWAY = { fossil: "Fossil", grey: "Grey, from natural gas", blue: "Blue, with carbon capture", green: "Green, from renewable power", e_fuel: "E-fuel, from renewable power", bio: "Bio", bio_lbg: "Bio-LNG" };
  const PATHWAY_SHORT = { fossil: "Fossil", grey: "Grey", blue: "Blue", green: "Green", e_fuel: "E-", bio: "Bio", bio_lbg: "Bio-" };
  const optionLabel = (fuel, pathway) => {
    if (fuel === "electricity") return "Electricity (grid)";
    if (fuel === "mgo" || fuel === "hfo") return FUEL_NAME[fuel];
    if (fuel === "lng") return pathway === "bio_lbg" ? "Bio-LNG" : "LNG";
    const p = PATHWAY_SHORT[pathway] ?? pathway;
    return p.endsWith("-") ? `${p}${fuel}` : `${p} ${fuel}`;
  };
  const VIEWS = ["home", "plan", "fuels", "calc", "ports", "tech"];
  const TOK = { ink: "#122036", ink2: "#4a5870", muted: "#8290a4", line: "#e4e9f1", grey: "#c9d0db" };
  const FONT = { family: "Poppins, \"Segoe UI\", system-ui, sans-serif", size: 12, color: TOK.ink2 };
  const CONFIG = { displayModeBar: false, responsive: true };
  const layout = (extra) => Object.assign({
    font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 56, r: 16, t: 10, b: 44 }, hovermode: "closest", hoverlabel: { font: { size: 12.5, family: FONT.family }, bgcolor: "#fff", bordercolor: P[200] },
    xaxis: { gridcolor: TOK.line, zerolinecolor: TOK.line }, yaxis: { gridcolor: TOK.line, zerolinecolor: TOK.line },
    legend: { orientation: "h", x: 1, xanchor: "right", y: 1.02, yanchor: "bottom", font: { size: 11.5 } },
  }, extra || {});
  const fmt = {
    int: (x) => Math.round(x).toLocaleString("en-IN"),
    cr: (inr) => "₹" + (inr / 1e7).toLocaleString("en-IN", { maximumFractionDigits: 1 }) + " cr",
    pct: (f, d = 0) => (Math.abs(f) * 100).toFixed(d) + "%",
    signed: (f, d = 0) => (f > 0.0005 ? "+" : f < -0.0005 ? "−" : "") + fmt.pct(f, d),
    sci: (x) => (x === 0 ? "0" : x.toExponential(2)),
    big: (x) => { if (x < 1e6) return fmt.int(x); const e = Math.floor(Math.log10(x)); return (x / 10 ** e).toFixed(1) + "×10" + String(e).replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]); },
    mt: (t) => (t >= 1e6 ? (t / 1e6).toFixed(2) + " million t" : Math.round(t).toLocaleString("en-IN") + " t"),
  };
  const toneCls = (f) => (f < -0.005 ? "better" : f > 0.005 ? "worse" : "");
  const vsDiesel = (f) => (f < -0.005 ? `${fmt.pct(f)} less CO₂ than diesel` : f > 0.005 ? `${fmt.pct(f)} more CO₂ than diesel` : "same CO₂ as diesel");
  const tile = (l, n, d) => `<div class="tile"><div class="tile-l">${l}</div><div class="tile-n">${n}</div>${d ? `<div class="tile-d">${d}</div>` : ""}</div>`;
  const stat = (l, n, d) => `<div><div class="tile-l">${l}</div><div class="tile-n">${n}</div>${d ? `<div class="tile-d">${d}</div>` : ""}</div>`;
  const chips = (mix) => Object.entries(mix).map(([f, n]) => `<span class="chip" style="${chipStyle(f)}">${n} × ${FUEL_NAME[f]}</span>`).join("");

  const api = async (path, body) => {
    const res = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { const d = data.detail; throw new Error(typeof d === "string" ? d : JSON.stringify(d)); }
    return data;
  };

  let META = null;
  const S = { job: null, result: null, selected: null, dirty: false, poll: null, calc: null, view: "home" };

  // ---- boot & routing -------------------------------------------------------------
  async function boot() {
    document.querySelectorAll(".link").forEach((b) => b.addEventListener("click", () => go(b.dataset.view)));
    document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => go(b.dataset.go)));
    window.addEventListener("hashchange", () => showView(viewFromHash()));
    showView(viewFromHash());
    try { META = await api("/api/meta"); } catch (err) { setStatus("bad", "offline"); showBanner("Planning service unreachable: " + err.message); return; }
    if (!META.model.loaded) showBanner("No trained model loaded: predictions are physics-only estimates.");
    setStatus("ok", "ready");
    buildHome(); buildPlan(); buildFuels(); buildCalc(); buildPorts(); buildTech();
    const params = new URLSearchParams(location.search);
    if (params.get("view") && !location.hash) go(params.get("view"));
    const linked = params.get("job");
    if (linked) { S.job = linked; $("progress").hidden = false; setStatus("busy", "loading"); S.poll = setInterval(pollJob, 250); }
    else runPlan();
  }
  const viewFromHash = () => { const v = location.hash.replace(/^#\/?/, ""); return VIEWS.includes(v) ? v : "home"; };
  const go = (name) => { if (location.hash !== "#/" + name) location.hash = "#/" + name; else showView(name); };
  function showView(name) {
    S.view = name;
    document.querySelectorAll(".link").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
    document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== "view-" + name; });
    window.scrollTo({ top: 0 });
    // Plotly sizes a chart when drawn; charts drawn while their page was hidden need a resize.
    requestAnimationFrame(() => document.querySelectorAll(`#view-${name} .js-plotly-plot`).forEach((el) => Plotly.Plots.resize(el)));
  }
  function setStatus(kind, text) { $("status-text").textContent = text; }
  function showBanner(text) { $("banner").hidden = false; $("banner").textContent = text; }
  function showError(id, text) { $(id).hidden = false; $(id).textContent = text; }

  // ---- home -----------------------------------------------------------------------
  function buildHome() {
    const m = META.model, held = m.held_out, hs = META.harit_sagar.targets;
    const n = Object.keys(META.ports).length;
    $("home-kpis").innerHTML =
      (m.loaded ? stat("Learned from", `${fmt.int(m.panel.n_ship_years)} <small>ship-years</small>`, `Real fuel reports from ${fmt.int(m.panel.n_ships)} ships, ${m.panel.reporting_periods[0]} to ${m.panel.reporting_periods.at(-1)}`) : stat("Fuel model", "Physics only", "No trained model is loaded")) +
      (held ? stat("Typical error", `${held.mape_pct.toFixed(0)}%`, `Tested on ${fmt.int(held.n_test_ships)} ships the model never saw`) : "") +
      stat("Harit Sagar target", `${fmt.pct(hs["2030"])} <small>by 2030</small>`, `and ${fmt.pct(hs["2047"])} less CO₂ by 2047, against ${META.harit_sagar.baseline_year}`) +
      stat("Ports covered", `${n}`, `Fuel and shore power availability for ${n} Indian ports`);
    $("home-cross").textContent = META.electrification_crossover_year;
    $("home-options").innerHTML = [0, 1, 2].map(() => `<div class="card" style="min-height:230px"><div class="skel" style="height:18px;width:40%"></div><div class="skel" style="height:30px;width:75%"></div><div class="skel" style="height:6px"></div><div class="skel" style="height:14px;width:60%"></div><div class="skel" style="height:22px;width:50%"></div></div>`).join("");
    if (m.loaded) $("foot-model").textContent = `Fuel model: ${m.name}, ${fmt.int(m.panel.n_ship_years)} ship-years`;
    const years = Object.keys(META.grid_factor_t_per_mwh).map(Number);
    gridChart("home-grid-chart", years);
  }
  function gridChart(id, years) {
    const y = years.map((yr) => META.grid_factor_t_per_mwh[yr]);
    Plotly.react(id, [{ type: "scatter", mode: "lines+markers", x: years, y, fill: "tozeroy", fillcolor: "rgba(58,116,196,0.10)", line: { color: P[500], width: 2.5 }, marker: { size: 5, color: years.map((yr) => (yr <= 2023 ? P[900] : P[400])) }, hovertemplate: "%{x}: %{y:.2f} t/MWh<extra></extra>" }],
      layout({ yaxis: { gridcolor: TOK.line, rangemode: "tozero" }, xaxis: { gridcolor: TOK.line }, showlegend: false, margin: { l: 40, r: 12, t: 10, b: 32 } }), CONFIG);
  }
  function renderHome(result) {
    const scen = $("scenario").selectedOptions[0]?.textContent ?? "";
    if (!result || !result.plans.length) { $("home-plan").innerHTML = `<div class="eyebrow">Recommended plan</div><div class="sub">No plan found for this fleet. Try other fuels on the Plan page.</div>`; $("home-options").innerHTML = ""; return; }
    const bal = byIdx(result.extremes.balanced ?? result.plans[0].index), d = dieselChange(bal), hs = bal.harit_sagar;
    $("home-plan").innerHTML = `<div class="eyebrow">Recommended plan · balanced</div>
      <div class="sub">${S.dirty ? "Your edited fleet" : scen}</div>
      <div class="plan-nums"><div><div class="big">${fmt.cr(bal.cost_inr)}</div><div class="plan-cap">cost per year</div></div><div><div class="big">${fmt.int(bal.emissions_tco2e)} t</div><div class="plan-cap">CO₂ per year</div></div></div>
      <div style="font-size:13px"><span class="${toneCls(d ?? 0)}">${vsDiesel(d ?? 0)}</span></div>
      <div class="chips">${chips(bal.fuel_mix)}</div>
      ${hs ? `<div><span class="badge ${hs.on_track ? "badge-on" : "badge-off"}">${hs.on_track ? "Meets" : "Misses"} Harit Sagar ${result.problem.year} target</span></div>` : ""}
      <div style="margin-top:auto"><button class="btn btn-sm" id="home-open-plan">See ship-by-ship plan</button></div>`;
    $("home-open-plan").addEventListener("click", () => { selectPlan(bal.index); go("plan"); });
    $("home-options-sub").textContent = `${result.plans.length} plans found for ${S.dirty ? "your edited fleet" : scen}`;
    $("home-options").innerHTML = HIGHLIGHTS.map((h) => planCard(h, byIdx(result.extremes[h.key]), true)).join("");
    document.querySelectorAll("#home-options .plan-card").forEach((el) => el.addEventListener("click", () => { selectPlan(Number(el.dataset.index)); go("plan"); }));
  }

  // ---- plan setup ---------------------------------------------------------------
  function buildPlan() {
    const picks = $("fuel-picks"), paths = $("pathways");
    Object.entries(META.fuels).forEach(([key, fuel]) => {
      if (fuel.is_energy_carrier) return;
      const b = document.createElement("button");
      b.type = "button"; b.className = "fuel-chip"; b.dataset.fuel = key; b.setAttribute("aria-pressed", "false");
      b.innerHTML = `<span class="box">✓</span><i class="swatch" style="background:${FUEL_COLOR[key]}"></i>${FUEL_NAME[key]}`;
      b.addEventListener("click", () => setPick(b, b.getAttribute("aria-pressed") !== "true"));
      picks.appendChild(b);
      const pathways = Object.keys(fuel.pathways);
      if (pathways.length > 1) {
        const l = document.createElement("label"); l.className = "field";
        l.innerHTML = `${FUEL_NAME[key]}<select data-pathway="${key}" class="input">${pathways.map((p) => `<option value="${p}">${PATHWAY[p] ?? p}</option>`).join("")}</select>`;
        paths.appendChild(l);
      }
    });
    $("scenario").addEventListener("change", loadScenario);
    $("year").addEventListener("change", fleetSummary);
    $("edit-fleet").addEventListener("click", () => $("editor-dialog").showModal());
    $("editor-done").addEventListener("click", () => $("editor-dialog").close());
    $("editor-dialog").addEventListener("close", fleetSummary);
    $("run").addEventListener("click", runPlan);
    $("add-vessel").addEventListener("click", () => { addVesselRow({}); markDirty(); });
    $("add-route").addEventListener("click", () => { addRouteRow({}); markDirty(); });
    $("download").addEventListener("click", downloadPlan);
    loadScenario();
  }
  const setPick = (b, on) => { b.setAttribute("aria-pressed", on ? "true" : "false"); b.classList.toggle("on", on); };
  const picked = () => [...document.querySelectorAll(".fuel-chip")].filter((b) => b.getAttribute("aria-pressed") === "true").map((b) => b.dataset.fuel);
  function markDirty() { S.dirty = true; fleetSummary(); }
  function loadScenario() {
    const spec = META.scenarios[$("scenario").value];
    $("year").value = String(spec.year); $("carbon").value = spec.carbon_price_inr_per_t; $("cap").value = "";
    document.querySelectorAll(".fuel-chip").forEach((b) => setPick(b, spec.fuels.includes(b.dataset.fuel)));
    document.querySelectorAll("[data-pathway]").forEach((s) => { if (spec.fuel_pathways[s.dataset.pathway]) s.value = spec.fuel_pathways[s.dataset.pathway]; });
    $("vessel-table").querySelector("tbody").innerHTML = ""; $("route-table").querySelector("tbody").innerHTML = "";
    spec.vessels.forEach(addVesselRow); spec.routes.forEach(addRouteRow);
    S.dirty = false; fleetSummary();
  }
  function fleetSummary() {
    const vessels = readVessels(), routes = readRoutes();
    const ports = [...new Set(routes.flatMap((r) => r.ports))].map((p) => (META.ports[p]?.display_name ?? p).replace(/ Port.*$/, "").replace(/ \(.*$/, ""));
    $("fleet-summary").innerHTML = `<span><b>${vessels.length}</b> ships</span><span><b>${routes.length}</b> routes between ${ports.join(", ")}</span><span><b>${fmt.mt(routes.reduce((a, r) => a + (r.annual_demand_t || 0), 0))}</b> of cargo a year</span>${S.dirty ? `<span class="badge badge-neutral">edited</span>` : ""}`;
  }
  const fleetCounts = () => { const v = readVessels(), r = readRoutes(); return { ships: v.length, routes: r.length, cargo: r.reduce((a, x) => a + (x.annual_demand_t || 0), 0) }; };
  const className = (key) => META.vessel_classes.find((c) => c.key === key)?.display_name ?? key;
  const classOptions = (sel) => META.vessel_classes.map((c) => `<option value="${c.key}" ${c.key === sel ? "selected" : ""}>${c.display_name}</option>`).join("");
  const portOptions = (sel) => Object.entries(META.ports).map(([k, p]) => `<option value="${k}" ${k === sel ? "selected" : ""}>${p.display_name}</option>`).join("");
  const td = (h) => `<td>${h}</td>`, num = (name, v, step = "any") => `<input type="number" step="${step}" data-f="${name}" value="${v ?? ""}">`;
  function addVesselRow(v) {
    const tr = document.createElement("tr");
    tr.innerHTML = td(`<input type="text" data-f="name" value="${v.name ?? "Ship-" + (Date.now() % 1000)}">`) + td(`<select data-f="vessel_class">${classOptions(v.vessel_class ?? "general_cargo")}</select>`) +
      td(num("capacity_t", v.capacity_t ?? 8000)) + td(num("design_efficiency_gco2_per_t_nmi", v.design_efficiency_gco2_per_t_nmi, "0.1")) + td(num("min_speed_kn", v.min_speed_kn ?? 8, "0.5")) + td(num("max_speed_kn", v.max_speed_kn ?? 14, "0.5")) +
      td(num("tank_volume_m3", v.tank_volume_m3 ?? 800)) + td(num("annual_cost_inr", v.annual_cost_inr ?? 8e7)) + td(num("available_hours", v.available_hours ?? 7000)) + td(`<button class="del" title="Remove">×</button>`);
    tr.querySelector(".del").addEventListener("click", () => { tr.remove(); markDirty(); }); tr.addEventListener("input", markDirty);
    $("vessel-table").querySelector("tbody").appendChild(tr);
  }
  function addRouteRow(r) {
    const tr = document.createElement("tr"); const ports = r.ports ?? ["mumbai", "cochin"];
    tr.innerHTML = td(`<input type="text" data-f="name" value="${r.name ?? "Route-" + (Date.now() % 1000)}">`) + td(`<select data-f="from">${portOptions(ports[0])}</select>`) + td(`<select data-f="to">${portOptions(ports[ports.length - 1])}</select>`) +
      td(num("distance_nmi", r.distance_nmi ?? 300)) + td(num("annual_demand_t", r.annual_demand_t ?? 200000)) + td(num("max_transit_hours", r.max_transit_hours)) + td(`<button class="del" title="Remove">×</button>`);
    tr.querySelector(".del").addEventListener("click", () => { tr.remove(); markDirty(); }); tr.addEventListener("input", markDirty);
    $("route-table").querySelector("tbody").appendChild(tr);
  }
  const readRows = (table) => [...table.querySelectorAll("tbody tr")].map((tr) => { const row = {}; tr.querySelectorAll("[data-f]").forEach((el) => { row[el.dataset.f] = el.type === "number" ? (el.value === "" ? null : Number(el.value)) : el.value; }); return row; });
  const readVessels = () => readRows($("vessel-table"));
  const readRoutes = () => readRows($("route-table")).map((r) => ({ name: r.name, ports: [r.from, r.to], distance_nmi: r.distance_nmi, annual_demand_t: r.annual_demand_t, max_transit_hours: r.max_transit_hours }));
  function buildRequest() {
    const fuels = picked(), pathways = {};
    document.querySelectorAll("[data-pathway]").forEach((s) => { if (fuels.includes(s.dataset.pathway)) pathways[s.dataset.pathway] = s.value; });
    const req = { scenario: S.dirty ? "custom" : $("scenario").value, year: Number($("year").value), fuels, fuel_pathways: pathways, carbon_price_inr_per_t: Number($("carbon").value || 0),
      emission_cap_tco2e: $("cap").value === "" ? null : Number($("cap").value), n_individuals: Number($("ind").value), n_generations: Number($("gen").value), seed: Number($("seed").value),
      compare_with: ["nsga2", "qbho", "greedy", "random"].filter((k) => $("cmp-" + k).checked) };
    if (req.scenario === "custom") { req.vessels = readVessels(); req.routes = readRoutes(); }
    return req;
  }
  async function runPlan() {
    $("plan-error").hidden = true;
    const req = buildRequest();
    if (!req.fuels.length) { showError("plan-error", "Select at least one fuel."); return; }
    let job;
    try { job = await api("/api/optimise", req); } catch (err) { showError("plan-error", err.message.replace(/^infeasible: cargo demand cannot be met even ignoring emissions and cost\. /, "This fleet cannot move that much cargo. ")); return; }
    S.job = job.job_id; $("run").disabled = true; $("run-compare").disabled = true;
    $("progress").hidden = false; $("fill").style.width = "0%"; $("progress-text").textContent = "Searching…";
    setStatus("busy", "searching"); S.poll = setInterval(pollJob, 250);
  }
  async function pollJob() {
    let s;
    try { s = await api(`/api/optimise/${S.job}`); } catch (err) { finishJob(); showError("plan-error", err.message); return; }
    if (s.progress) {
      const p = s.progress, w = (100 * p.generation / p.n_generations).toFixed(1) + "%", text = `${fmt.int(p.n_evaluations)} plans checked, ${p.archive_size} kept so far`;
      $("fill").style.width = w; $("progress-text").textContent = text;
      if ($("home-fill")) { $("home-fill").style.width = w; $("home-progress").textContent = p.generation >= p.n_generations ? "Comparing with other methods…" : text; }
    }
    if (s.status === "running" || s.status === "queued") return;
    finishJob(); applyRequest(s.request);
    if (s.status === "failed") { showError("plan-error", s.error); return; }
    if (s.result) renderResult(s.result);
  }
  function finishJob() { clearInterval(S.poll); S.poll = null; $("run").disabled = false; $("run-compare").disabled = false; $("progress").hidden = true; setStatus("ok", "ready"); }
  function applyRequest(req) {
    if (!req) return;
    if (req.scenario !== "custom") $("scenario").value = req.scenario;
    const spec = META.scenarios[req.scenario];
    $("year").value = String(req.year ?? spec?.year ?? 2030); $("carbon").value = req.carbon_price_inr_per_t ?? spec?.carbon_price_inr_per_t ?? 0; $("cap").value = req.emission_cap_tco2e ?? "";
    $("ind").value = req.n_individuals; $("gen").value = req.n_generations; $("seed").value = req.seed;
    if (req.fuels) document.querySelectorAll(".fuel-chip").forEach((b) => setPick(b, req.fuels.includes(b.dataset.fuel)));
    if (req.fuel_pathways) document.querySelectorAll("[data-pathway]").forEach((s) => { if (req.fuel_pathways[s.dataset.pathway]) s.value = req.fuel_pathways[s.dataset.pathway]; });
    ["nsga2", "qbho", "greedy", "random"].forEach((k) => { $("cmp-" + k).checked = (req.compare_with || []).includes(k); });
    fleetSummary();
  }

  // ---- results ------------------------------------------------------------------
  const HIGHLIGHTS = [
    { key: "cheapest", title: "Lowest cost", desc: "Spend the least each year", color: TOK.muted },
    { key: "balanced", title: "Balanced", desc: "A good cut in CO₂ for a fair price", color: P[500] },
    { key: "cleanest", title: "Lowest emissions", desc: "Cut the most CO₂", color: P[800] },
  ];
  const byIdx = (i) => S.result.plans.find((p) => p.index === i);
  const dieselChange = (plan) => (plan.harit_sagar ? -plan.harit_sagar.achieved_reduction : null);
  const dominantFuel = (plan) => Object.entries(plan.fuel_mix).sort((a, b) => b[1] - a[1])[0]?.[0];
  const titleOf = (plan) => HIGHLIGHTS.find((h) => S.result.extremes[h.key] === plan.index)?.title ?? `Plan ${plan.index + 1}`;

  function renderResult(result) {
    S.result = result;
    if (!result.plans.length) { showError("plan-error", "No plan delivers all the cargo with these fuels in this year: a fuel is not yet available on one of the routes."); $("results").hidden = true; $("plan-empty").hidden = false; renderHome(result); return; }
    $("results").hidden = false; $("plan-empty").hidden = true;
    const c = fleetCounts();
    $("results-sub").textContent = `${result.plans.length} plans deliver all ${fmt.mt(c.cargo)} of cargo in ${result.problem.year}`;
    $("plan-cards").innerHTML = HIGHLIGHTS.map((h) => planCard(h, byIdx(result.extremes[h.key]))).join("");
    document.querySelectorAll("#plan-cards .plan-card").forEach((el) => el.addEventListener("click", () => selectPlan(Number(el.dataset.index))));
    drawTradeoff(result);
    selectPlan(result.extremes.balanced ?? result.plans[0].index);
    renderTech(result);
    renderHome(result);
  }
  const highlightMax = () => { const ps = HIGHLIGHTS.map((h) => byIdx(S.result.extremes[h.key])).filter(Boolean); return { cost: Math.max(...ps.map((p) => p.cost_inr)), co2: Math.max(...ps.map((p) => p.emissions_tco2e)) }; };
  const meter = (f, cls = "") => `<div class="meter ${cls}"><i style="width:${Math.max(3, 100 * f).toFixed(1)}%"></i></div>`;
  function planCard(h, plan, compact) {
    const d = dieselChange(plan), hs = plan.harit_sagar, mx = highlightMax();
    return `<button class="plan-card" data-index="${plan.index}">
      <div class="plan-top"><span class="plan-title"><i class="swatch" style="background:${h.color};border-radius:50%"></i>${h.title}</span>${compact ? "" : `<span class="sel-tag badge badge-on">Selected</span>`}</div>
      <div class="plan-desc">${h.desc}</div>
      <div class="plan-nums"><div><div class="plan-num">${fmt.cr(plan.cost_inr)}</div><div class="plan-cap">cost per year</div>${meter(plan.cost_inr / mx.cost)}</div><div><div class="plan-num">${fmt.int(plan.emissions_tco2e)} t</div><div class="plan-cap">CO₂ per year</div>${meter(plan.emissions_tco2e / mx.co2, "co2")}</div></div>
      <div style="font-size:13px;color:var(--ink2)"><span class="${toneCls(d ?? 0)}">${vsDiesel(d ?? 0)}</span> · ${plan.n_deployed} of ${plan.assignments.length} ships sail</div>
      <div class="chips" style="gap:6px">${chips(plan.fuel_mix)}</div>
      ${hs ? `<div><span class="badge ${hs.on_track ? "badge-on" : "badge-off"}">${hs.on_track ? "Meets" : "Misses"} Harit Sagar ${S.result.problem.year}</span></div>` : ""}</button>`;
  }
  function drawTradeoff(result) {
    const plans = result.plans, ex = result.extremes;
    const label = (p) => HIGHLIGHTS.filter((h) => ex[h.key] === p.index).map((h) => h.title).join(" · ");
    const tech = result.baselines.length > 0;
    const traces = [{
      type: "scatter", mode: "markers+text", name: "our plans", x: plans.map((p) => p.cost_inr / 1e7), y: plans.map((p) => p.emissions_tco2e),
      text: plans.map(label), textfont: { size: 11, color: TOK.ink, family: FONT.family }, textposition: plans.map((p) => (ex.cleanest === p.index ? "top left" : ex.cheapest === p.index ? "top right" : "bottom center")),
      marker: { size: plans.map((p) => (label(p) ? 16 : 11)), color: plans.map((p) => FUEL_COLOR[dominantFuel(p)] ?? P[500]), line: { color: "#fff", width: 1.5 } },
      customdata: plans.map((p) => [p.index, Object.entries(p.fuel_mix).map(([f, n]) => `${n} ${FUEL_NAME[f]}`).join(" · "), fmt.signed(dieselChange(p) ?? 0)]),
      hovertemplate: "₹%{x:.1f} cr · %{y:,.0f} t CO₂<br>%{customdata[2]} vs diesel · %{customdata[1]}<extra></extra>",
    }];
    const NAME = { nsga2: "NSGA-II", qbho: "QBHO", greedy: "greedy", random_search: "random" };
    if (tech) result.baselines.forEach((b) => { if (b.front.length) traces.push({ type: "scatter", mode: "markers", name: NAME[b.algorithm] ?? b.algorithm, visible: "legendonly", x: b.front.map((r) => r[2] / 1e7), y: b.front.map((r) => r[1]), marker: { size: 8, symbol: "diamond-open", color: TOK.muted, line: { width: 1.5, color: TOK.muted } }, hovertemplate: `${b.algorithm}<br>₹%{x:.1f} cr · %{y:,.0f} t<extra></extra>` }); });
    Plotly.react("tradeoff-chart", traces, layout({ xaxis: { title: { text: "cost, ₹ crore per year" }, gridcolor: TOK.line }, yaxis: { title: { text: "CO₂, tonnes per year" }, gridcolor: TOK.line, rangemode: "tozero" }, showlegend: tech, margin: { l: 60, r: 16, t: tech ? 40 : 10, b: 44 } }), CONFIG);
    $("tradeoff-chart").removeAllListeners?.("plotly_click");
    $("tradeoff-chart").on("plotly_click", (ev) => { const pt = ev.points[0]; if (pt.curveNumber === 0) selectPlan(pt.customdata[0]); });
    const used = [...new Set(plans.map(dominantFuel))].filter(Boolean);
    $("fuel-legend").innerHTML = used.map((f) => `<span><i class="swatch" style="background:${FUEL_COLOR[f]};border-radius:50%"></i>${FUEL_NAME[f]}</span>`).join("");
  }
  function selectPlan(index) {
    const plan = byIdx(index) || S.result.plans[0]; S.selected = plan;
    document.querySelectorAll("#plan-cards .plan-card").forEach((el) => el.classList.toggle("selected", Number(el.dataset.index) === plan.index));
    const d = dieselChange(plan), hs = plan.harit_sagar, year = S.result.problem.year;
    $("detail-title").textContent = titleOf(plan);
    $("detail-sub").innerHTML = `${fmt.cr(plan.cost_inr)} a year · ${fmt.int(plan.emissions_tco2e)} t CO₂ a year · <span class="${toneCls(d ?? 0)}">${fmt.signed(d ?? 0)} vs diesel</span>` + (hs ? ` · ${hs.on_track ? "meets" : "misses"} Harit Sagar ${year} (${fmt.pct(hs.achieved_reduction)} cut, ${fmt.pct(hs.required_reduction)} needed)` : "");
    const deployed = plan.assignments.filter((a) => a.deployed), idle = plan.assignments.filter((a) => !a.deployed);
    $("assign-table").querySelector("tbody").innerHTML = deployed.map((a) => `<tr><td class="td"><b>${a.name}</b><br><span class="sub">${className(a.vessel_class)}</span></td><td class="td">${a.route.replace("-", " to ")}</td><td class="td"><span class="chip" style="${chipStyle(a.fuel)}" title="${PATHWAY[a.pathway] ?? ""}">${FUEL_NAME[a.fuel]}</span></td><td class="td-r">${a.speed_kn.toFixed(1)} kn</td></tr>`).join("") +
      (idle.length ? `<tr><td class="td sub" style="white-space:normal;border-bottom:0" colspan="4">${idle.length} in reserve: ${idle.map((a) => a.name).join(", ")}</td></tr>` : "");
    const plans = S.result.plans;
    if ($("tradeoff-chart").data) Plotly.restyle("tradeoff-chart", { "marker.line.color": [plans.map((p) => (p.index === plan.index ? TOK.ink : "#fff"))], "marker.line.width": [plans.map((p) => (p.index === plan.index ? 3 : 1.5))] }, [0]);
  }
  function downloadPlan() {
    if (!S.selected) return;
    const blob = new Blob([JSON.stringify({ plan: S.selected, problem: S.result.problem, energy_source: S.result.energy_source, generated: new Date().toISOString() }, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `fleet_plan_${S.selected.index + 1}.json`; a.click(); URL.revokeObjectURL(a.href);
  }

  // ---- how it works ---------------------------------------------------------------
  function buildTech() {
    const m = META.model, held = m.held_out;
    $("run-compare").addEventListener("click", () => { ["nsga2", "qbho", "greedy"].forEach((k) => { $("cmp-" + k).checked = true; }); $("tech-sub").textContent = "Running the comparison, about half a minute…"; runPlan(); });
    if (m.loaded) $("how-1").textContent = `A machine learning model (${m.name}) trained on ${fmt.int(m.panel.n_ship_years)} real ship fuel reports predicts the energy each ship needs at each speed.`;
    $("model-kpis").innerHTML = held
      ? stat("Average error", `${held.mape_pct.toFixed(1)}%`, "mean absolute percentage error on unseen ships") + stat("Median error", `${held.median_ape_pct.toFixed(1)}%`, "half of predictions are closer than this") +
        stat("Unseen test ships", fmt.int(held.n_test_ships), "no ship appears in both training and test") + stat("Range coverage", fmt.pct(held.interval_coverage_80), "of real values fall in the 80% likely range")
      : "";
  }
  function renderTech(result) {
    const nsga = result.baselines.find((b) => b.algorithm.toLowerCase().includes("nsga"));
    $("tech-sub").textContent = `${result.problem.n_vessels} ships, ${result.problem.n_routes} routes, ${result.problem.year}`;
    $("tech-kpis").innerHTML = stat("Plans checked", fmt.int(result.n_evaluations), `in ${result.elapsed_s.toFixed(1)} seconds`) + stat("Possible plans", fmt.big(result.problem.search_space_size), `${result.problem.n_bits} Q-bits of choices`) +
      stat("Quality vs NSGA-II", nsga ? (nsga.hypervolume > 0 ? (result.hypervolume / nsga.hypervolume).toFixed(2) + "×" : "∞") : "n/a", nsga ? `hypervolume; ${result.plans.length} vs ${nsga.n_plans} plans` : "NSGA-II not run") + stat("Repairs", result.repairs, "impossible plans fixed during search");
    $("tech-charts").hidden = false;
    const h = result.history, gens = h.archive_size.map((_, i) => i + 1);
    Plotly.react("progress-chart", [
      { type: "scatter", mode: "lines", name: "plans kept", x: gens, y: h.archive_size, line: { color: P[600], width: 2.5 }, yaxis: "y" },
      { type: "scatter", mode: "lines", name: "Q-bit diversity", x: gens, y: h.diversity, line: { color: P[300], width: 2.5 }, yaxis: "y2" },
    ], layout({ grid: { rows: 2, columns: 1, pattern: "independent" }, xaxis: { title: { text: "generation" }, gridcolor: TOK.line, anchor: "y2" }, yaxis: { domain: [0.56, 1], rangemode: "tozero", gridcolor: TOK.line }, yaxis2: { domain: [0, 0.42], range: [0, 1.05], gridcolor: TOK.line }, margin: { l: 40, r: 16, t: 30, b: 40 } }), CONFIG);
    $("methods-card").hidden = !result.baselines.length; $("tech-charts").classList.toggle("g2", result.baselines.length > 0);
    if (result.baselines.length) {
      const NAMES = { nsga2: "NSGA-II", qbho: "QBHO", greedy: "Greedy", random_search: "Random" };
      const rows = [{ algorithm: "QIEA + QPSO (ours)", hypervolume: result.hypervolume }, ...result.baselines.map((b) => ({ ...b, algorithm: NAMES[b.algorithm] ?? b.algorithm }))];
      Plotly.react("methods-chart", [{ type: "bar", orientation: "h", x: rows.map((r) => r.hypervolume), y: rows.map((r) => r.algorithm), marker: { color: rows.map((_, i) => (i ? P[200] : P[600])) }, text: rows.map((r) => fmt.sci(r.hypervolume)), textposition: "outside", cliponaxis: false, hovertemplate: "%{y}: %{x:.3e}<extra></extra>" }],
        layout({ xaxis: { gridcolor: TOK.line, exponentformat: "e", range: [0, Math.max(...rows.map((r) => r.hypervolume)) * 1.3] }, yaxis: { automargin: true, autorange: "reversed" }, margin: { l: 10, r: 20, t: 8, b: 30 }, showlegend: false }), CONFIG);
    }
    const p = result.problem;
    $("provenance").hidden = false;
    $("provenance").innerHTML = `<b>Where the numbers come from.</b> ${result.energy_source}. ${p.n_vessels} vessels × ${p.n_routes} routes × ${p.fuels.length} fuels = ${p.n_bits} Q-bits. Fuel pathways: ${Object.entries(p.fuel_pathways).map(([f, w]) => `${FUEL_NAME[f] ?? f} ${PATHWAY[w] ?? w}`).join(", ")}. Carbon price ₹${fmt.int(p.carbon_price_inr_per_t)}/t${p.emission_cap_tco2e ? `, cap ${fmt.int(p.emission_cap_tco2e)} t` : ""}. Harit Sagar baseline: the same deployment on diesel. <a href="?job=${S.job}#/plan">Link to this result</a>.`;
    if (S.view === "tech") showView("tech");
  }

  // ---- fuels --------------------------------------------------------------------
  function buildFuels() { ["fuel-preset", "fuel-year", "funnel-only"].forEach((id) => $(id).addEventListener("change", runFuels)); runFuels(); drawElectrification(); }
  const verdictColor = (x) => (x.fuel === "mgo" ? P[900] : x.verdict_flips ? P[200] : x.vs_baseline < -0.005 ? P[500] : x.vs_baseline > 0.005 ? TOK.grey : TOK.muted);
  async function runFuels() {
    const tj = Number($("fuel-preset").value), year = Number($("fuel-year").value);
    const r = await api("/api/emissions/compare", { shaft_energy_mj: tj * 1e6, year }).catch((e) => { showBanner(e.message); return null; });
    if (!r) return;
    const rows = r.rows.slice().sort((a, b) => a.total_t - b.total_t), diesel = r.rows.find((x) => x.fuel === "mgo"), funnel = $("funnel-only").checked;
    const names = rows.map((x) => optionLabel(x.fuel, x.pathway) + "  ");
    const traces = [{ type: "bar", orientation: "h", name: "whole lifecycle", x: rows.map((x) => x.total_t), y: names, marker: { color: rows.map(verdictColor) },
      text: rows.map((x) => (x.fuel === "mgo" ? "diesel today" : x.verdict_flips ? `${fmt.signed(x.vs_baseline)}, looks clean at the funnel` : fmt.signed(x.vs_baseline))), textposition: "outside", cliponaxis: false, textfont: { size: 12, family: FONT.family, color: TOK.ink }, hovertemplate: "%{y}: %{x:,.0f} t CO₂e<extra></extra>" }];
    if (funnel) traces.push({ type: "scatter", mode: "markers", name: "funnel-only tool reports", x: rows.map((x) => x.stack_only_t), y: names, marker: { symbol: "line-ns-open", size: 20, color: TOK.ink, line: { width: 3 } }, hovertemplate: "funnel-only: %{x:,.0f} t<extra></extra>" });
    Plotly.react("fuel-chart", traces, layout({ xaxis: { title: { text: "tonnes of CO₂ per year, whole lifecycle" }, gridcolor: TOK.line, range: [0, Math.max(...rows.map((x) => x.total_t)) * 1.3] }, yaxis: { automargin: true, autorange: "reversed", tickfont: { size: 12.5 } }, margin: { l: 6, r: 12, t: 6, b: 44 }, showlegend: funnel,
      shapes: [{ type: "line", x0: diesel.total_t, x1: diesel.total_t, yref: "paper", y0: 0, y1: 1, line: { color: P[900], dash: "dot", width: 1.5 } }] }), CONFIG);
    $("fuel-sub").textContent = `${tj} TJ · ${year} · grid ${r.grid_factor_t_per_mwh.toFixed(2)} t/MWh`;
    const flips = rows.filter((x) => x.verdict_flips), lng = rows.find((x) => x.fuel === "lng"), elec = rows.find((x) => x.fuel === "electricity");
    const facts = [
      { l: "Look clean, are dirtier", n: `${flips.length} fuels`, d: `${flips.map((x) => optionLabel(x.fuel, x.pathway)).join(", ")}: no carbon at the funnel, but made from natural gas, so more CO₂ than diesel overall.` },
      { l: "LNG with measured methane slip", n: `<span class="${toneCls(lng?.vs_baseline ?? 0)}">${fmt.signed(lng?.vs_baseline ?? 0)}</span>`, d: "1.72% slip measured on 745 real ships. With the 3.1% regulatory default, LNG would be worse than diesel." },
      { l: "Electric tug on India's grid", n: `<span class="${toneCls(elec?.vs_baseline ?? 0)}">${fmt.signed(elec?.vs_baseline ?? 0)}</span> <small style="font-size:13px;color:var(--muted)">in ${year}</small>`, d: `Beats diesel on CO₂ from <b>${META.electrification_crossover_year}</b>, and removes NOx and soot at the quayside from day one.` },
    ];
    $("fuel-facts").innerHTML = facts.map((f) => tile(f.l, f.n, f.d)).join("");
    $("home-facts-sub").textContent = `one harbour tug-sized demand, ${year}`;
    if (tj === 12 || !$("home-facts").innerHTML) $("home-facts").innerHTML = facts.map((f) => `<div class="fact"><div class="tile-l">${f.l}</div><div class="n">${f.n}</div><div class="d">${f.d}</div></div>`).join("");
    $("fuel-table").querySelector("tbody").innerHTML = r.rows.map((x) => `<tr><td class="td">${optionLabel(x.fuel, x.pathway)}</td><td class="td-r">${fmt.int(x.total_t)}</td><td class="td-r">${fmt.int(x.breakdown.well_to_tank)}</td><td class="td-r">${fmt.int(x.breakdown.combustion_co2)}</td><td class="td-r">${fmt.int(x.breakdown.methane_slip)}</td><td class="td-r">${fmt.int(x.breakdown.n2o)}</td><td class="td-r">${fmt.int(x.breakdown.pilot_fuel)}</td><td class="td-r">${fmt.signed(x.vs_baseline)}</td><td class="td-r">${fmt.signed(x.stack_only_vs_baseline)}</td><td class="td">${x.confidence}</td></tr>`).join("");
  }
  async function drawElectrification() {
    const e = await api("/api/emissions/electrification").catch(() => null); if (!e) return;
    Plotly.react("elec-chart", [
      { type: "scatter", mode: "lines", name: "battery-electric", x: e.years, y: e.electric_t_per_tj, line: { color: P[500], width: 3 }, hovertemplate: "%{x}: %{y:.0f} t/TJ<extra></extra>" },
      { type: "scatter", mode: "lines", name: "diesel", x: e.years, y: e.years.map(() => e.baseline_t_per_tj), line: { color: P[900], dash: "dash", width: 2 }, hoverinfo: "skip" },
    ], layout({ yaxis: { gridcolor: TOK.line, rangemode: "tozero" }, xaxis: { gridcolor: TOK.line }, margin: { l: 40, r: 12, t: 30, b: 36 },
      annotations: e.crossover_year ? [{ x: e.crossover_year, y: e.baseline_t_per_tj, text: `electric wins from ${e.crossover_year}`, showarrow: true, arrowhead: 2, arrowcolor: P[700], ax: 0, ay: -34, font: { size: 12, color: P[900], family: FONT.family } }] : [] }), CONFIG);
  }

  // ---- calculator ---------------------------------------------------------------
  function buildCalc() {
    $("calc-class").innerHTML = META.vessel_classes.map((c) => `<option value="${c.key}" ${c.key === "bulk_carrier" ? "selected" : ""}>${c.display_name}</option>`).join("");
    $("calc-fuel").innerHTML = ["mgo", "lng", "methanol", "ammonia", "hydrogen"].map((f) => `<option value="${f}">${FUEL_NAME[f]}</option>`).join("");
    $("calc-class").addEventListener("change", () => { const c = META.vessel_classes.find((x) => x.key === $("calc-class").value); if (c && !c.in_mrv_scope) $("calc-design").value = ""; fetchCurve(); });
    $("calc-design").addEventListener("change", fetchCurve); $("calc-fuel").addEventListener("change", drawCalc); $("calc-speed").addEventListener("input", drawCalc);
    fetchCurve();
  }
  async function fetchCurve() {
    $("calc-error").hidden = true;
    const design = $("calc-design").value, base = { vessel_class: $("calc-class").value, design_efficiency_gco2_per_t_nmi: design === "" ? null : Number(design) };
    let first;
    try { first = await api("/api/predict", base); } catch (err) { showError("calc-error", err.message); return; }
    const ref = first.reference_speed_kn, lo = Math.max(2, Math.round(ref * 0.3 * 2) / 2), hi = Math.round(ref * 1.25 * 2) / 2;
    const speeds = []; for (let v = lo; v <= hi + 1e-9; v += 0.5) speeds.push(Number(v.toFixed(1)));
    try { S.calc = await api("/api/predict", { ...base, speeds_kn: speeds }); } catch (err) { showError("calc-error", err.message); return; }
    const slider = $("calc-speed"); slider.min = lo; slider.max = hi; slider.step = 0.5;
    if (Number(slider.value) < lo || Number(slider.value) > hi) slider.value = ref;
    drawCalc();
  }
  function drawCalc() {
    const r = S.calc; if (!r) return;
    const fuel = $("calc-fuel").value, speed = Number($("calc-speed").value);
    $("calc-speed-label").textContent = speed.toFixed(1) + " kn";
    const pts = r.points, at = (v) => pts.reduce((a, b) => (Math.abs(b.speed_kn - v) < Math.abs(a.speed_kn - v) ? b : a));
    const cur = at(speed), slow = at(Math.max(Number($("calc-speed").min), speed - 2)), mass = (p) => p.fuel_t_per_1000_nmi[fuel];
    const band = (p) => [mass(p) * p.p10_energy_per_nmi_mj / p.energy_per_nmi_mj, mass(p) * p.p90_energy_per_nmi_mj / p.energy_per_nmi_mj];
    const best = pts.reduce((a, b) => (mass(b) < mass(a) ? b : a)), [lo, hi] = band(cur);
    const row = (l, v) => `<div class="row"><span>${l}</span><span>${v}</span></div>`;
    $("calc-answer").innerHTML = `<div class="tile-l">${FUEL_NAME[fuel]} per 1,000 nautical miles</div><div class="answer-n">${mass(cur).toFixed(0)} t</div><div class="sub" style="margin-bottom:6px">likely between ${lo.toFixed(0)} and ${hi.toFixed(0)} t at ${speed.toFixed(1)} knots</div>` +
      row("Engine power", `${fmt.int(cur.power_kw)} kW`) + (slow.speed_kn < cur.speed_kn ? row(`Slow down to ${slow.speed_kn.toFixed(1)} kn`, `<span class="better">${fmt.pct(1 - mass(slow) / mass(cur))} less fuel</span> per mile`) : "") + row("Most efficient speed", `${best.speed_kn.toFixed(1)} kn`) + row("Typical service speed", `${r.reference_speed_kn.toFixed(1)} kn`) +
      (r.cold_start ? `<div><span class="badge badge-off">Estimate: no certificate, so the range is wide</span></div>` : `<div><span class="badge badge-on">Based on certificate · about ${r.model.held_out ? r.model.held_out.mape_pct.toFixed(0) : "20"}% typical error</span></div>`);
    $("calc-sub").textContent = `${r.vessel_class_display}`;
    const x = pts.map((p) => p.speed_kn);
    Plotly.react("calc-chart", [
      { type: "scatter", mode: "lines", x, y: pts.map((p) => band(p)[1]), line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { type: "scatter", mode: "lines", name: "likely range", x, y: pts.map((p) => band(p)[0]), fill: "tonexty", fillcolor: "rgba(58,116,196,0.14)", line: { width: 0 }, hoverinfo: "skip" },
      { type: "scatter", mode: "lines", name: "estimate", x, y: pts.map(mass), line: { color: P[600], width: 3 }, hovertemplate: "%{x:.1f} kn · %{y:.0f} t<extra></extra>" },
      { type: "scatter", mode: "markers+text", name: "selected", x: [cur.speed_kn], y: [mass(cur)], text: [`${mass(cur).toFixed(0)} t`], textposition: "top left", textfont: { family: FONT.family, color: P[900], size: 12 }, marker: { color: P[900], size: 12, line: { color: "#fff", width: 2 } }, hoverinfo: "skip" },
    ], layout({ xaxis: { title: { text: "speed, knots" }, gridcolor: TOK.line }, yaxis: { title: { text: "tonnes per 1,000 nmi" }, gridcolor: TOK.line, rangemode: "tozero" }, showlegend: false }), CONFIG);
    const held = r.model.held_out;
    $("calc-provenance").innerHTML = `<b>Model.</b> ${r.model.source}. ` + (held ? `Tested on ${fmt.int(held.n_test_ships)} unseen ships: average error ${held.mape_pct.toFixed(1)}%, median ${held.median_ape_pct.toFixed(1)}%, R² ${held.r2.toFixed(2)}. ` : "") + `Reference ${fmt.int(r.reference_intensity_mj_per_nmi)} MJ/nmi at ${r.reference_speed_kn.toFixed(1)} kn; hotel load ${fmt.pct(r.aux_share)}; power grows with the cube of speed.`;
  }

  // ---- ports --------------------------------------------------------------------
  const AVAIL = { available: "Yes", limited: "Limited", planned: "Planned", none: "No" };
  function buildPorts() {
    $("ports-year").addEventListener("change", drawPorts); drawPorts();
    gridChart("grid-chart", Object.keys(META.grid_factor_t_per_mwh).map(Number));
  }
  function drawPorts() {
    const year = Number($("ports-year").value), fuels = Object.keys(META.fuels).filter((f) => f !== "electricity");
    const yearKey = [2024, 2030, 2040, 2047].filter((y) => y <= year).pop() ?? 2024;
    const ic = (v) => `<span class="avail av-${v}">${AVAIL[v] ?? v}</span>`;
    $("ports-table").querySelector("thead").innerHTML = `<tr><th>Port</th><th class="c">Shore power</th>${fuels.map((f) => `<th class="c">${FUEL_NAME[f]}</th>`).join("")}</tr>`;
    $("ports-table").querySelector("tbody").innerHTML = Object.entries(META.ports).map(([k, p]) => {
      const tags = [p.is_demo_site ? "pilot site" : "", p.gttp_port ? "green tugs" : "", p.green_hydrogen_hub ? "H₂ hub" : ""].filter(Boolean).map((t) => `<span class="chip-soft">${t}</span>`).join(" ");
      return `<tr><td class="td" style="white-space:normal"><b>${p.display_name}</b> <span class="sub">${p.state ?? ""}</span> ${tags}</td><td class="td c">${ic(p.shore_power[yearKey])}</td>${fuels.map((f) => `<td class="td c">${ic(p.fuel_availability[f][yearKey])}</td>`).join("")}</tr>`;
    }).join("");
    const all = Object.values(META.ports), n = all.length, has = (v) => ["available", "limited"].includes(v);
    const count = (f) => all.filter((p) => has(p.fuel_availability[f][yearKey])).length;
    const clean = ["lng", "methanol", "ammonia", "hydrogen"].reduce((a, f) => a + count(f), 0);
    $("ports-kpis").innerHTML = stat("Ports", n, "in the planner") + stat("Shore power", `${all.filter((p) => has(p.shore_power[yearKey])).length} <small>of ${n}</small>`, `ports can plug ships in by ${year}`) +
      stat("LNG bunkering", `${count("lng")} <small>of ${n}</small>`, `ports supply LNG by ${year}`) + stat("Clean fuel points", clean, `methanol, ammonia, hydrogen and LNG supply points by ${year}`);
    $("ports-note").textContent = `In ${year}: ` + fuels.map((f) => `${FUEL_NAME[f]} at ${count(f)} of ${n} ports`).join(" · ") + `. Availability years are placeholders to confirm with each port authority.`;
  }

  window.addEventListener("error", (e) => showBanner("Page error: " + e.message));
  boot();
})();
