"""Viewports — how a *human* watches a Mirage scene (the agent drives via the API
and inspects via rendered PNGs).

Two interchangeable frontends over the same scene/sim:

* ``WebViewport`` — writes a self-contained three.js page (loaded from a CDN) that
  reconstructs the scene's primitives and optionally plays back a simulation
  trajectory. Cross-platform and screenshot-friendly; serve it locally.
* ``launch_native`` — opens MuJoCo's native OpenGL viewer (interactive; needs a
  display).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

# three.js page: a root group is rotated -90deg about X so Mirage's Z-up maps to
# screen-up; objects are then placed directly in Mirage coordinates.
_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>html,body{margin:0;height:100%;background:#0e0f12;overflow:hidden}
#hud{position:fixed;left:10px;top:8px;color:#aab;font:12px monospace}</style></head>
<body><div id="hud">Mirage web viewport — drag to orbit · scroll to zoom</div>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0e0f12);
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.01, 1000);
camera.position.set(3, 2.2, 3);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.4, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 0.9); sun.position.set(2, 5, 3); scene.add(sun);
const root = new THREE.Group(); root.rotation.x = -Math.PI/2; scene.add(root);  // Mirage Z-up
const grid = new THREE.GridHelper(20, 40, 0x335, 0x223); grid.rotation.x = Math.PI/2; root.add(grid);
const meshes = {};
function col(c){ return new THREE.Color(c[0], c[1], c[2]); }
function build(scn){
  for (const [name, e] of Object.entries(scn.entities||{})){
    const g = e.geometry||{}, p = (g.params||{}), m = e.material||{base_color:[.7,.7,.75,1]};
    let geo;
    if (g.kind==='box'){ const s=p.size||[1,1,1]; geo=new THREE.BoxGeometry(s[0],s[1],s[2]); }
    else if (g.kind==='sphere'){ geo=new THREE.SphereGeometry(p.radius||0.5, 32, 24); }
    else if (g.kind==='cylinder'){ geo=new THREE.CylinderGeometry(p.radius||0.5,p.radius||0.5,p.height||1,32); geo.rotateX(Math.PI/2); }
    else if (g.kind==='plane'){ const s=p.size||[10,10]; geo=new THREE.BoxGeometry(s[0]*2,s[1]*2,0.02); }
    else { geo=new THREE.BoxGeometry(0.3,0.3,0.3); }
    const mat=new THREE.MeshStandardMaterial({color:col(m.base_color||[.7,.7,.75]), metalness:m.metallic||0, roughness:(m.roughness==null?0.6:m.roughness)});
    const mesh=new THREE.Mesh(geo,mat);
    const t=e.transform||{}; const pos=t.position||[0,0,0]; mesh.position.set(pos[0],pos[1],pos[2]);
    const q=t.rotation||[1,0,0,0]; mesh.quaternion.set(q[1],q[2],q[3],q[0]);
    root.add(mesh); meshes[name]=mesh;
  }
}
let frames=[], fi=0;
function animate(){
  requestAnimationFrame(animate);
  if (frames.length){ const f=frames[fi%frames.length]; for(const[n,pos]of Object.entries(f)){ if(meshes[n]) meshes[n].position.set(pos[0],pos[1],pos[2]); } fi++; }
  controls.update(); renderer.render(scene, camera);
}
function load(n){ return (window.__DATA__&&window.__DATA__[n]!==undefined)?Promise.resolve(window.__DATA__[n]):fetch(n).then(r=>r.ok?r.json():[]); }
load('scene.json').then(scn=>{ build(scn); load('frames.json').then(f=>{ frames=f||[]; }).catch(()=>{}); animate(); });
addEventListener('resize',()=>{ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); });
</script></body></html>
"""


def trajectory_from_sim(sim, scene, steps: int, dt: float = 1.0 / 60.0) -> list:
    """Roll out the sim, recording every dynamic entity's position per frame
    (for ``WebViewport`` playback)."""
    names = [n for n in scene.entity_names() if scene.physics_kind(n) == "dynamic"]
    frames = []
    for _ in range(int(steps)):
        sim.step_for(dt)
        frames.append({n: [float(v) for v in sim.body_pos(n)] for n in names})
    return frames


class WebViewport:
    """Write/serve a three.js page that renders a scene (+ optional trajectory)."""

    def __init__(self, scene, frames=None, title: str = "Mirage"):
        self.scene_dict = scene.to_dict() if hasattr(scene, "to_dict") else scene
        self.frames = frames or []
        self.title = title

    def write(self, out_dir, inline: bool = False) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        scene_json = json.dumps(self.scene_dict)
        frames_json = json.dumps(self.frames)
        (out / "scene.json").write_text(scene_json, encoding="utf-8")
        (out / "frames.json").write_text(frames_json, encoding="utf-8")
        html = _HTML.replace("__TITLE__", self.title)
        if inline:  # embed the data so the page opens straight from disk (no server)
            inject = f'<script>window.__DATA__={{"scene.json":{scene_json},"frames.json":{frames_json}}};</script>\n'
            html = html.replace("<script src=", inject + "<script src=", 1)
        (out / "index.html").write_text(html, encoding="utf-8")
        return out / "index.html"

    def serve(self, out_dir, port: int = 0, open_browser: bool = False) -> str:
        """Write the page to ``out_dir`` and serve it on localhost; returns the URL."""
        import http.server
        import functools
        self.write(out_dir)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
        url = f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        self._httpd = httpd
        return url


def launch_native(sim, passive: bool = False):
    """Open MuJoCo's native interactive viewer for a ``MujocoSim`` (needs a display).
    ``passive=True`` returns immediately (non-blocking); otherwise it blocks until
    the window closes."""
    import mujoco.viewer
    if passive:
        return mujoco.viewer.launch_passive(sim.model, sim.data)
    return mujoco.viewer.launch(sim.model, sim.data)
