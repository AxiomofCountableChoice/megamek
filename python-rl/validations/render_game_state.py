import os
import sys
import torch
import json
import math

def generate_interactive_html(traj_path, output_path="render.html"):
    if not os.path.exists(traj_path):
        print(f"File not found: {traj_path}")
        return
        
    print(f"Loading trajectory: {traj_path}")
    traj_data = torch.load(traj_path, map_location='cpu', weights_only=False)
    
    topology = traj_data.get("topology_payload", {})
    steps = traj_data.get("steps", [])
    
    json_data = {
        "topology": topology,
        "steps": []
    }
    
    all_reports = []
    total_reward = 0.0
    for s in steps:
        raw = s.get("raw_payload", {})
        state = raw.get("state", {})
        reward = s.get("reward", 0.0)
        total_reward += reward
        
        reports = raw.get("reports", [])
        all_reports.extend(reports)
        
        action_dict = s.get("action_dict", {})
        clean_action = {}
        for k, v in action_dict.items():
            if torch.is_tensor(v):
                clean_action[k] = v.item() if v.numel() == 1 else v.tolist()
            else:
                clean_action[k] = v
                
        step_dict = {
            "context": raw.get("context", "UNKNOWN"),
            "action": clean_action,
            "reward": reward,
            "total_reward": total_reward,
            "reports": raw.get("reports", []),
            "entities": state.get("entities", []),
            "weapons": state.get("weapons", []),
            "los_threats": state.get("los_threat_edges", []),
            "los_targets": state.get("los_target_edges", []),
            "movement_threats": state.get("movement_threat_edges", []),
            "move_tmm_0": state.get("move_type_tmm_0_edges", []),
            "move_tmm_1": state.get("move_type_tmm_1_edges", []),
            "move_tmm_2": state.get("move_type_tmm_2_edges", []),
            "move_tmm_3": state.get("move_type_tmm_3_edges", []),
            "move_tmm_4": state.get("move_type_tmm_4_edges", []),
            "partial_covers": state.get("partial_cover_edges", []),
            "entities_meta": state.get("entities_meta", []),
            "mask": raw.get("mask", {})
        }
        print(f"Step {len(json_data['steps'])} context: {step_dict['context']} TMM0 edges: {len(step_dict['move_tmm_0'])} TMM1 edges: {len(step_dict['move_tmm_1'])} los_threats: {len(step_dict['los_threats'])} movement_threats: {len(step_dict['movement_threats'])}")
        json_data["steps"].append(step_dict)
        
    json_data["all_reports"] = all_reports
    json_string = json.dumps(json_data)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MegaMek Trajectory Viewer</title>
    <style>
        body {{ font-family: sans-serif; display: flex; flex-direction: column; height: 100vh; margin: 0; background: #1e1e1e; color: #ddd; }}
        #header {{ background: #333; padding: 10px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444; }}
        #main {{ display: flex; flex: 1; overflow: hidden; }}
        #map-container {{ flex: 1; overflow: auto; position: relative; background: #2a2a2a; padding: 20px; }}
        #sidebar {{ width: 350px; background: #333; display: flex; flex-direction: column; border-left: 1px solid #444; }}
        .panel {{ padding: 10px; border-bottom: 1px solid #444; }}
        .panel-title {{ font-weight: bold; margin-bottom: 5px; color: #fff; border-bottom: 1px solid #555; padding-bottom: 3px; }}
        #reports-panel {{ flex: 1; overflow-y: auto; font-size: 0.85em; padding: 10px; }}
        .report-line {{ padding: 2px 0; border-bottom: 1px solid #3a3a3a; white-space: pre-wrap; font-family: monospace; font-size: 0.95em; }}
        button {{ background: #555; color: white; border: none; padding: 5px 15px; cursor: pointer; }}
        button:hover {{ background: #666; }}
        button:disabled {{ background: #444; color: #777; cursor: not-allowed; }}
        
        svg {{ background: #222; }}
        .hex {{ stroke: #444; stroke-width: 1; transition: fill 0.2s; }}
        .hex-woods {{ fill: #2d5a27; }}
        .hex-water {{ fill: #1c5277; }}
        .hex-swamp {{ fill: #4a345a; }}
        .hex-clear {{ fill: #3a3a3a; }}
        .hex-impassable {{ fill: #111; }}
        .unit {{ stroke: #fff; stroke-width: 1.5; cursor: pointer; }}
        .unit-friendly {{ fill: #2b70c9; }}
        .unit-enemy {{ fill: #c92b2b; }}
        .unit-text {{ fill: white; font-size: 10px; font-weight: bold; text-anchor: middle; dominant-baseline: central; pointer-events: none; }}
        .facing-line {{ stroke: yellow; stroke-width: 2; pointer-events: none; }}
        .los-threat {{ stroke: orange; stroke-width: 2; stroke-dasharray: 4; }}
        .los-target {{ stroke: red; stroke-width: 3; }}
        .move-threat {{ stroke: #c92b2b; stroke-width: 2; stroke-dasharray: 3; opacity: 0.7; }}
        .move-tmm-0 {{ stroke: #555; stroke-width: 2; stroke-dasharray: 2; opacity: 0.8; }}
        .move-tmm-1 {{ stroke: #2b70c9; stroke-width: 2; stroke-dasharray: 2; opacity: 0.8; }}
        .move-tmm-2 {{ stroke: #2bc970; stroke-width: 2; stroke-dasharray: 2; opacity: 0.8; }}
        .move-tmm-3 {{ stroke: #c9a72b; stroke-width: 2; stroke-dasharray: 2; opacity: 0.8; }}
        .move-tmm-4 {{ stroke: #c9522b; stroke-width: 2; stroke-dasharray: 2; opacity: 0.8; }}
        .legend-item {{ cursor: pointer; user-select: none; }}
        .legend-item:hover {{ background: #444; }}
        .legend-item.disabled {{ opacity: 0.3; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        th, td {{ text-align: left; padding: 2px 5px; border-bottom: 1px solid #444; }}
        th {{ color: #aaa; font-weight: normal; }}
    </style>
</head>
<body>
    <div id="header">
        <div>
            <h2 style="margin: 0; font-size: 1.2em;">MegaMek Trajectory Viewer</h2>
        </div>
        <div>
            <button id="btn-prev">Prev</button>
            <span id="step-label" style="margin: 0 15px; font-weight: bold;">Step 0 / 0</span>
            <button id="btn-next">Next</button>
        </div>
    </div>
    <div id="main">
        <div id="map-container">
            <svg id="map" width="100%" height="100%"></svg>
        </div>
        <div id="sidebar">
            <div class="panel">
                <div class="panel-title">Context</div>
                <div style="margin-bottom: 5px;">Context: <strong id="meta-context"></strong></div>
                <div style="color: #bbccff; margin-bottom: 5px; font-size: 0.9em;">Action: <span id="meta-action"></span></div>
                <div style="color: #88ff88;">Step Reward: <span id="meta-reward"></span></div>
                <div style="color: #88ff88;">Total Reward: <span id="meta-total-reward"></span></div>
            </div>
            <div class="panel" id="entity-panel" style="display: none;">
                <div class="panel-title">Selected Unit</div>
                <div id="entity-details"></div>
            </div>
            <div class="panel" style="padding-bottom: 5px;">
                <div class="panel-title">Legend</div>
                <div style="font-size: 0.85em;">
                    <div class="legend-item" onclick="toggleEdge('los-target')"><span style="color:#ff4f4f; font-weight:bold;">&#x2501;&#x2501;&#x2501;</span> LOS Target</div>
                    <div class="legend-item" onclick="toggleEdge('los-threat')"><span style="color:orange; font-weight:bold;">- - -</span> LOS Threat</div>
                    <div class="legend-item" onclick="toggleEdge('move-threat')"><span style="color:#c92b2b; font-weight:bold;">- &nbsp;- &nbsp;-</span> Movement Threat</div>
                    <div class="legend-item" onclick="toggleEdge('move-tmm-0')"><span style="color:#555; font-weight:bold;">- - -</span> Move TMM 0</div>
                    <div class="legend-item" onclick="toggleEdge('move-tmm-1')"><span style="color:#2b70c9; font-weight:bold;">- - -</span> Move TMM 1</div>
                    <div class="legend-item" onclick="toggleEdge('move-tmm-2')"><span style="color:#2bc970; font-weight:bold;">- - -</span> Move TMM 2</div>
                    <div class="legend-item" onclick="toggleEdge('move-tmm-3')"><span style="color:#c9a72b; font-weight:bold;">- - -</span> Move TMM 3</div>
                    <div class="legend-item" onclick="toggleEdge('move-tmm-4')"><span style="color:#c9522b; font-weight:bold;">- - -</span> Move TMM 4</div>
                    <div class="legend-item" onclick="toggleEdge('partial-cover')"><span style="color:#a04fff; font-weight:bold;">. . .</span> Partial Cover</div>
                </div>
                <div class="panel-title" style="margin-top: 10px; border: none;">Hexes</div>
                <div style="font-size: 0.85em;">
                    <span style="display:inline-block; width:12px; height:12px; background:#3a3a3a; margin-right:5px;"></span> Clear<br>
                    <span style="display:inline-block; width:12px; height:12px; background:#2d5a27; margin-right:5px;"></span> Woods<br>
                    <span style="display:inline-block; width:12px; height:12px; background:#1c5277; margin-right:5px;"></span> Water<br>
                    <span style="display:inline-block; width:12px; height:12px; background:#4a345a; margin-right:5px;"></span> Swamp<br>
                    <span style="display:inline-block; width:12px; height:12px; background:#111; margin-right:5px;"></span> Impassable
                </div>
            </div>
            <div class="panel-title" style="padding: 10px 10px 0 10px; border: none;">Game Reports</div>
            <div id="reports-panel"></div>
        </div>
    </div>
    
    <script>
        const traj = {json_string};
        const width = traj.topology.width || 16;
        const height = traj.topology.height || 16;
        const hexes = traj.topology.hex_nodes || [];
        const steps = traj.steps || [];
        
        const repPanel = document.getElementById("reports-panel");
        repPanel.innerHTML = (traj.all_reports && traj.all_reports.length > 0) ? 
            traj.all_reports.map(r => `<div class="report-line">${{r}}</div>`).join('') :
            '<div style="color: #666; font-style: italic;">No reports</div>';
        repPanel.scrollTop = repPanel.scrollHeight;
        
        const HEX_SIZE = 30;
        const HEX_WIDTH = HEX_SIZE * 2;
        const HEX_HEIGHT = HEX_SIZE * 1.732;
        const SVG_PAD = 50;
        
        let currentStep = 0;
        let selectedUnitIdx = -1;
        
        let edgeVisibility = {{
            'los-target': true,
            'los-threat': true,
            'move-threat': true,
            'move-tmm-0': true,
            'move-tmm-1': true,
            'move-tmm-2': true,
            'move-tmm-3': true,
            'move-tmm-4': true,
            'partial-cover': true
        }};
        
        function toggleEdge(className) {{
            edgeVisibility[className] = !edgeVisibility[className];
            renderStep();
            document.querySelectorAll('.legend-item').forEach(el => {{
                if (el.getAttribute('onclick').includes(className)) {{
                    if (edgeVisibility[className]) el.classList.remove('disabled');
                    else el.classList.add('disabled');
                }}
            }});
        }}
        
        const svg = document.getElementById("map");
        
        function initSvg() {{
            const totalW = width * HEX_WIDTH * 0.75 + HEX_WIDTH * 0.25 + SVG_PAD*2;
            const totalH = height * HEX_HEIGHT + HEX_HEIGHT * 0.5 + SVG_PAD*2;
            svg.setAttribute("viewBox", `0 0 ${{totalW}} ${{totalH}}`);
        }}
        
        function drawHexGrid() {{
            svg.innerHTML = '';
            
            for(let y=0; y<height; y++) {{
                for(let x=0; x<width; x++) {{
                    let idx = y * width + x;
                    let hexFeat = hexes[idx] || Array(14).fill(0);
                    // 0:elev, 1:woods, 2:water, 3:pavement, 4:building, 5:rough, 6:rubble, 7:swamp, 8:mud, 9:ice, 10:snow, 11:fire, 12:smoke, 13:impassable
                    
                    let className = "hex hex-clear";
                    if (hexFeat[13] > 0) className = "hex hex-impassable";
                    else if (hexFeat[1] > 0) className = "hex hex-woods";
                    else if (hexFeat[2] > 0) className = "hex hex-water";
                    else if (hexFeat[7] > 0) className = "hex hex-swamp";
                    
                    let px = SVG_PAD + x * HEX_WIDTH * 0.75;
                    let py = SVG_PAD + y * HEX_HEIGHT + (x % 2 !== 0 ? HEX_HEIGHT/2 : 0);
                    
                    let points = [];
                    for(let i=0; i<6; i++) {{
                        let angle_rad = i * Math.PI / 3;
                        points.push(`${{px + HEX_SIZE * Math.cos(angle_rad)}},${{py + HEX_SIZE * Math.sin(angle_rad)}}`);
                    }}
                    
                    let poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
                    poly.setAttribute("points", points.join(" "));
                    poly.setAttribute("class", className);
                    svg.appendChild(poly);
                    
                    if (hexFeat[0] > 0) {{
                        let txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
                        txt.setAttribute("x", px);
                        txt.setAttribute("y", py - HEX_SIZE/2 + 5);
                        txt.setAttribute("fill", "#888");
                        txt.setAttribute("font-size", "10");
                        txt.setAttribute("text-anchor", "middle");
                        txt.textContent = hexFeat[0];
                        svg.appendChild(txt);
                    }}
                }}
            }}
        }}
        
        function getPxPy(x, y) {{
            let px = SVG_PAD + x * HEX_WIDTH * 0.75;
            let py = SVG_PAD + y * HEX_HEIGHT + (x % 2 !== 0 ? HEX_HEIGHT/2 : 0);
            return [px, py];
        }}
        
        function renderStep() {{
            document.getElementById("step-label").innerText = `Step ${{currentStep}} / ${{steps.length - 1}}`;
            document.getElementById("btn-prev").disabled = currentStep === 0;
            document.getElementById("btn-next").disabled = currentStep === steps.length - 1;
            
            const stepData = steps[currentStep];
            if (!stepData) return;
            
            document.getElementById("meta-context").innerText = stepData.context;
            document.getElementById("meta-action").innerText = JSON.stringify(stepData.action || {{}});
            document.getElementById("meta-reward").innerText = stepData.reward.toFixed(4);
            document.getElementById("meta-total-reward").innerText = stepData.total_reward.toFixed(4);
            
            // Reports are now statically populated in the panel globally.
            
            drawHexGrid();
            
            if (stepData.context === "MOVEMENT_INFERENCE") {{
                const drawEdgeToHex = (edges, className) => {{
                    if (!edgeVisibility[className]) return;
                    if (!edges) return;
                    edges.forEach(t => {{
                        if (selectedUnitIdx !== -1 && t[0] !== selectedUnitIdx) return;
                        let u = stepData.entities[t[0]];
                        if (!u) return;
                        let uPos = getPxPy(u[1], u[2]);
                        let hx = t[1] % width;
                        let hy = Math.floor(t[1] / width);
                        let hPos = getPxPy(hx, hy);
                        let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                        line.setAttribute("x1", uPos[0]); line.setAttribute("y1", uPos[1]);
                        line.setAttribute("x2", hPos[0]); line.setAttribute("y2", hPos[1]);
                        line.setAttribute("class", className);
                        svg.appendChild(line);
                    }});
                }};
                drawEdgeToHex(stepData.los_threats, "los-threat");
                drawEdgeToHex(stepData.movement_threats, "move-threat");
                drawEdgeToHex(stepData.move_tmm_0, "move-tmm-0");
                drawEdgeToHex(stepData.move_tmm_1, "move-tmm-1");
                drawEdgeToHex(stepData.move_tmm_2, "move-tmm-2");
                drawEdgeToHex(stepData.move_tmm_3, "move-tmm-3");
                drawEdgeToHex(stepData.move_tmm_4, "move-tmm-4");
            }}
            
            if (stepData.los_targets && edgeVisibility['los-target']) {{
                stepData.los_targets.forEach(t => {{
                    if (selectedUnitIdx !== -1 && t[0] !== selectedUnitIdx && t[1] !== selectedUnitIdx) return;
                    let u1 = stepData.entities[t[0]];
                    let u2 = stepData.entities[t[1]];
                    if(!u1 || !u2) return;
                    let uPos1 = getPxPy(u1[1], u1[2]);
                    let uPos2 = getPxPy(u2[1], u2[2]);
                    
                    let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                    line.setAttribute("x1", uPos1[0]); line.setAttribute("y1", uPos1[1]);
                    line.setAttribute("x2", uPos2[0]); line.setAttribute("y2", uPos2[1]);
                    line.setAttribute("class", "los-target");
                    svg.appendChild(line);
                }});
            }}
            
            if (stepData.entities) {{
                stepData.entities.forEach((u, i) => {{
                    if (u[1] < 0 || u[2] < 0) return; // Not on board
                    let isFriendly = u[0] > 0.5;
                    let pos = getPxPy(u[1], u[2]);
                    
                    let g = document.createElementNS("http://www.w3.org/2000/svg", "g");
                    
                    let circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    circle.setAttribute("cx", pos[0]);
                    circle.setAttribute("cy", pos[1]);
                    circle.setAttribute("r", 14);
                    circle.setAttribute("class", isFriendly ? "unit unit-friendly" : "unit unit-enemy");
                    
                    if (selectedUnitIdx === i) {{
                        circle.setAttribute("stroke", "yellow");
                        circle.setAttribute("stroke-width", "3");
                    }}
                    
                    circle.onclick = () => {{
                        selectedUnitIdx = selectedUnitIdx === i ? -1 : i;
                        updateEntityPanel();
                        renderStep();
                    }};
                    g.appendChild(circle);
                    
                    let sinF = u[3];
                    let cosF = u[4];
                    let fLen = 22;
                    let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                    line.setAttribute("x1", pos[0]); line.setAttribute("y1", pos[1]);
                    line.setAttribute("x2", pos[0] + sinF * fLen); line.setAttribute("y2", pos[1] - cosF * fLen); 
                    line.setAttribute("class", "facing-line");
                    g.appendChild(line);
                    
                    let txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
                    txt.setAttribute("x", pos[0]);
                    txt.setAttribute("y", pos[1]);
                    txt.setAttribute("class", "unit-text");
                    txt.textContent = i;
                    g.appendChild(txt);
                    
                    svg.appendChild(g);
                }});
            }}
            
            // Draw Weapon Attack Lasers
            if ((stepData.context === "WEAPON_INFERENCE" || stepData.context === "WEAPON_BC") && stepData.mask && stepData.mask.valid_targets && stepData.action) {{
                let activeId = stepData.mask.active_entity;
                let activeIdx = -1;
                if (stepData.entities_meta) {{
                    stepData.entities_meta.forEach((meta, idx) => {{
                        if (meta && meta.id === activeId) activeIdx = idx;
                    }});
                }}
                
                if (activeIdx !== -1 && stepData.entities[activeIdx]) {{
                    let shooterPos = getPxPy(stepData.entities[activeIdx][1], stepData.entities[activeIdx][2]);
                    
                    let firedTargets = new Set();
                    for (let key in stepData.action) {{
                        if (key.startsWith("weapon_") && stepData.action[key] !== -1) {{
                            firedTargets.add(stepData.action[key]);
                        }}
                    }}
                    
                    firedTargets.forEach(tgtNodeIdx => {{
                        let vt = stepData.mask.valid_targets[tgtNodeIdx];
                        if (vt && stepData.entities[vt.target_entity_index]) {{
                            let targetIdx = vt.target_entity_index;
                            let targetPos = getPxPy(stepData.entities[targetIdx][1], stepData.entities[targetIdx][2]);
                            
                            let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                            line.setAttribute("x1", shooterPos[0]); line.setAttribute("y1", shooterPos[1]);
                            line.setAttribute("x2", targetPos[0]); line.setAttribute("y2", targetPos[1]);
                            line.setAttribute("stroke", "#ff3333");
                            line.setAttribute("stroke-width", "5");
                            line.setAttribute("stroke-dasharray", "8,4");
                            line.setAttribute("style", "filter: drop-shadow(0 0 5px #ff0000); pointer-events: none;");
                            svg.appendChild(line);
                        }}
                    }});
                }}
            }}
            
            updateEntityPanel();
        }}
        
        function updateEntityPanel() {{
            const p = document.getElementById("entity-panel");
            if (selectedUnitIdx < 0 || !steps[currentStep].entities || !steps[currentStep].entities[selectedUnitIdx]) {{
                p.style.display = "none";
                return;
            }}
            p.style.display = "block";
            const u = steps[currentStep].entities[selectedUnitIdx];
            
            let locs = ["HD", "CT", "RT", "LT", "RA", "LA", "RL", "LL"];
            
            let statusHtml = [];
            if(u[17]) statusHtml.push('PRONE');
            if(u[18]) statusHtml.push('DESTROYED');
            if(u[19]) statusHtml.push('IMMOBILE');
            
            let html = `
                <div style="font-weight:bold; font-size: 1.1em; margin-bottom: 5px; color:${{u[0]>0.5 ? '#2b70c9' : '#c92b2b'}}">Unit ${{selectedUnitIdx}} (${{u[0]>0.5 ? 'Friendly' : 'Enemy'}})</div>
                <div style="color:orange; font-weight:bold;">${{statusHtml.join(' | ')}}</div>
                <table style="margin-top: 5px;">
                    <tr><th>Heat</th><td>${{u[5]}} / ${{u[6]}}</td></tr>
                    <tr><th>MP</th><td>W:${{u[13]}} R:${{u[14]}} J:${{u[15]}}</td></tr>
                    <tr><th>TMM</th><td>${{u[9]}}</td></tr>
                    <tr><th>Crew</th><td>G:${{u[11]}} P:${{u[12]}}</td></tr>
                    <tr><th>Stats</th><td>${{u[16]}}t, H:${{u[20]}}</td></tr>
                </table>
                
                <h4 style="margin:10px 0 5px 0">Armor / Structure</h4>
                <table>
                    <tr><th>Loc</th><th>Armor</th><th>Struct</th></tr>
            `;
            for(let i=0; i<8; i++) {{
                let a = (u[21+i] * 100).toFixed(0);
                let ra = (u[29+i] * 100).toFixed(0);
                let s = (u[37+i] * 100).toFixed(0);
                html += `<tr><td>${{locs[i]}}</td><td>${{a}}%</td><td>${{ra}}%</td><td>${{s}}%</td></tr>`;
            }}
            html += `</table>`;
            
            if (steps[currentStep].entities_meta && steps[currentStep].entities_meta[selectedUnitIdx]) {{
                let meta = steps[currentStep].entities_meta[selectedUnitIdx];
                if (meta.weapons && meta.weapons.length > 0) {{
                    html += `<h4 style="margin:10px 0 5px 0">Weapons</h4><ul style="margin:0; padding-left:20px; font-size:0.9em;">`;
                    meta.weapons.forEach(w => {{ html += `<li>${{w}}</li>`; }});
                    html += `</ul>`;
                }}
            }}
            
            let stepData = steps[currentStep];
            if (stepData.context === "WEAPON_INFERENCE" && stepData.mask && stepData.mask.valid_targets) {{
                let isShooter = stepData.entities_meta && stepData.entities_meta[selectedUnitIdx] && stepData.entities_meta[selectedUnitIdx].id === stepData.mask.active_entity;
                if (isShooter) {{
                    html += `<h4 style="margin:10px 0 5px 0; color: orange;">Targeting Probabilities</h4>`;
                    stepData.mask.valid_targets.forEach(t => {{
                        let tgtName = `Target ${{t.target_entity_index}}`;
                        if (stepData.entities_meta && stepData.entities_meta[t.target_entity_index]) {{
                            tgtName = stepData.entities_meta[t.target_entity_index].name;
                        }}
                        html += `<div style="font-size:0.85em; margin-bottom:5px;"><strong>${{tgtName}}</strong><br>`;
                        t.valid_weapons.forEach(w => {{
                            let th = w.to_hit;
                            let htmlTh = th <= 12 ? `<span style="color:#88ff88">${{th}}+</span>` : `<span style="color:#ff8888">N/A</span>`;
                            html += `&bull; ${{w.weapon_name}}: ${{htmlTh}}<br>`;
                        }});
                        html += `</div>`;
                    }});
                }}
            }}
            
            document.getElementById("entity-details").innerHTML = html;
        }}
        
        document.getElementById("btn-prev").onclick = () => {{ if(currentStep > 0) {{ currentStep--; renderStep(); }} }};
        document.getElementById("btn-next").onclick = () => {{ if(currentStep < steps.length-1) {{ currentStep++; renderStep(); }} }};
        
        initSvg();
        renderStep();
    </script>
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Interactive HTML saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_game_state.py <path_to_trajectory.pt> [output.html]")
        sys.exit(1)
        
    traj = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "render.html"
    generate_interactive_html(traj, out)
