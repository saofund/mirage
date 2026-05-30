import pytest

mujoco = pytest.importorskip("mujoco")  # skip suite if the [mujoco] extra isn't installed

from mirage import Scene, Entity, Transform, Geometry, PhysicsBody, Camera, Engine
from mirage.mujoco_backend import MujocoSim, MujocoPhysics, MujocoRenderer, scene_to_mjcf


def _ground_and_box(z=1.0) -> Scene:
    s = Scene(name="t")
    s.add(Entity(name="ground", geometry=Geometry(kind="plane", params={"size": [5, 5]}),
                 physics=PhysicsBody(kind="static")))
    s.add(Entity(name="box", transform=Transform(position=[0, 0, z]),
                 geometry=Geometry(kind="box", params={"size": [0.4, 0.4, 0.4]}),
                 physics=PhysicsBody(kind="dynamic")))
    return s


def test_scene_to_mjcf_parses():
    xml = scene_to_mjcf(_ground_and_box())
    assert "<mujoco" in xml and "freejoint" in xml
    mujoco.MjModel.from_xml_string(xml)  # must be valid MJCF


def test_from_scene_collision_and_rest():
    sim = MujocoSim.from_scene(_ground_and_box(1.0))
    sim.step_for(1.5)
    assert sim.body_pos("box")[2] == pytest.approx(0.2, abs=0.05)  # rests at half-height


def test_render_modalities_shapes_nonblank():
    sim = MujocoSim.from_scene(_ground_and_box(0.3))
    imgs = sim.render(160, 120, modalities=("rgb", "depth", "segmentation"))
    assert imgs["rgb"].shape == (120, 160, 3)
    assert imgs["depth"].shape == (120, 160)
    assert imgs["rgb"].mean() > 1  # not all black


def test_hinge_joint_moves():
    xml = ('<mujoco><option gravity="0 0 -9.81"/><worldbody>'
           '<body pos="0 0 1"><joint name="h" type="hinge" axis="0 1 0"/>'
           '<geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.03"/></body>'
           '</worldbody></mujoco>')
    sim = MujocoSim.from_mjcf(xml)
    a0 = float(sim.joint("h").qpos[0])
    sim.step_for(0.5)
    assert abs(float(sim.joint("h").qpos[0]) - a0) > 0.1  # swings under gravity


def test_initial_velocity_honored():
    s = Scene(name="v")
    s.add(Entity(name="ground", geometry=Geometry(kind="plane", params={"size": [5, 5]}),
                 physics=PhysicsBody(kind="static")))
    s.add(Entity(name="ball", transform=Transform(position=[0, 0, 0.5]),
                 geometry=Geometry(kind="sphere", params={"radius": 0.1}),
                 physics=PhysicsBody(kind="dynamic", linear_velocity=[3.0, 0.0, 0.0])))
    sim = MujocoSim.from_scene(s)
    sim.step_for(0.3)
    assert sim.body_pos("ball")[0] > 0.3  # moved along +x from its initial velocity


def test_engine_with_mujoco_backends_syncs_usd():
    s = _ground_and_box(1.2)
    s.add(Camera(name="cam", width=64, height=48))
    sim = MujocoSim.from_scene(s)
    eng = Engine(scene=s, physics=MujocoPhysics(sim=sim), renderer=MujocoRenderer(sim=sim))
    eng.step(dt=0.1, steps=15)
    assert s.get_position("box")[2] < 1.2  # fell, and was synced back into the USD scene
    rr = eng.render("cam")
    assert rr.data["rgb"].shape == (48, 64, 3)


_OBJ_CUBE = """
v -0.1 -0.1 -0.1
v 0.1 -0.1 -0.1
v 0.1 0.1 -0.1
v -0.1 0.1 -0.1
v -0.1 -0.1 0.1
v 0.1 -0.1 0.1
v 0.1 0.1 0.1
v -0.1 0.1 0.1
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""


def test_mesh_asset_import(tmp_path):
    obj = tmp_path / "cube.obj"
    obj.write_text(_OBJ_CUBE)
    s = Scene(name="m")
    s.add(Entity(name="ground", geometry=Geometry(kind="plane", params={"size": [5, 5]}),
                 physics=PhysicsBody(kind="static")))
    s.add(Entity(name="part", transform=Transform(position=[0, 0, 1.0]),
                 geometry=Geometry(kind="mesh", params={"path": str(obj).replace("\\", "/"), "scale": [1, 1, 1]}),
                 physics=PhysicsBody(kind="dynamic", mass=0.5)))
    sim = MujocoSim.from_scene(s)
    z0 = sim.body_pos("part")[2]
    sim.step_for(1.0)
    assert sim.body_pos("part")[2] < z0  # the imported mesh fell under gravity
    assert sim.render(120, 90, modalities=("rgb",))["rgb"].shape == (90, 120, 3)


def test_from_urdf_string_joint_moves():
    urdf = """<robot name="r">
 <link name="base"><inertial><origin xyz="0 0 0"/><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link>
 <link name="l1"><inertial><origin xyz="0.25 0 0"/><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
   <visual><origin xyz="0.25 0 0"/><geometry><box size="0.5 0.1 0.1"/></geometry></visual>
   <collision><origin xyz="0.25 0 0"/><geometry><box size="0.5 0.1 0.1"/></geometry></collision></link>
 <joint name="j1" type="continuous"><parent link="base"/><child link="l1"/><axis xyz="0 1 0"/><origin xyz="0 0 0.5"/></joint>
</robot>"""
    sim = MujocoSim.from_urdf(urdf)
    assert sim.model.njnt >= 1
    a0 = float(sim.data.qpos[0])
    sim.step_for(0.5)
    assert abs(float(sim.data.qpos[0]) - a0) > 0.05  # hinge swings under gravity
