#!/usr/bin/env python3
"""Offline smoke test for nut_detector + nut_tracker LOGIC.

Runs on a plain dev machine (no ROS) by stubbing rclpy / the message packages /
cv2, importing the REAL node modules, bypassing their __init__ (so no Node
machinery is needed), and exercising the high-risk pure logic:

  * geometry round-trip: project a known floor nut INTO the image with the real
    OAK-D intrinsics + the real captured TF, then run the node's project_and_gate
    and confirm it recovers the same map coordinate (millimetre level).
  * each detector gate: physical-size, range, depth-on-floor.
  * depth_to_metres + image_to_bgr decoders.
  * tracker: nearest-neighbour de-dup, min_hits confirmation, drive-over
    collection, collected-position freeze, marker colours.

This validates maths/logic, NOT ROS plumbing (topics/QoS/TF live) — that needs
the robot. Exit code 0 = all passed.
"""

import os
import sys
import math
import types
import struct

import numpy as np

# ---------------------------------------------------------------------------
# 1. Stub every ROS / cv2 import the node modules pull in, BEFORE importing them.
# ---------------------------------------------------------------------------

def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

# rclpy + submodules
_mod("rclpy").init = lambda *a, **k: None
sys.modules["rclpy"].shutdown = lambda *a, **k: None
sys.modules["rclpy"].spin = lambda *a, **k: None

node_mod = _mod("rclpy.node")
class _Node:
    def __init__(self, *a, **k):
        pass
node_mod.Node = _Node

dur_mod = _mod("rclpy.duration")
class _Duration:
    def __init__(self, *a, **k):
        pass
dur_mod.Duration = _Duration

time_mod = _mod("rclpy.time")
class _Time:
    def __init__(self, *a, **k):
        pass
    @classmethod
    def from_msg(cls, msg):
        return cls()
time_mod.Time = _Time

qos_mod = _mod("rclpy.qos")
class _QoSProfile:
    def __init__(self, *a, **k):
        self.depth = k.get("depth", 1)
        self.durability = None
        self.history = None
class _Enum:
    TRANSIENT_LOCAL = 1
    KEEP_LAST = 1
qos_mod.QoSProfile = _QoSProfile
qos_mod.DurabilityPolicy = _Enum
qos_mod.HistoryPolicy = _Enum

# message packages
def _simple(name, attrs):
    cls = type(name, (), {"__init__": lambda self: [setattr(self, a, v()) for a, v in attrs.items()] and None})
    return cls

class _Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""
class _Vec:
    def __init__(self):
        self.x = 0.0; self.y = 0.0; self.z = 0.0; self.w = 1.0
class _Pose:
    def __init__(self):
        self.position = _Vec(); self.orientation = _Vec()
class _PoseArray:
    def __init__(self):
        self.header = _Header(); self.poses = []
class _Image:
    def __init__(self):
        self.header = _Header(); self.height = 0; self.width = 0
        self.encoding = ""; self.is_bigendian = 0; self.step = 0; self.data = b""
class _CameraInfo:
    def __init__(self):
        self.k = [0.0] * 9; self.width = 0; self.height = 0; self.header = _Header()
class _String:
    def __init__(self):
        self.data = ""
class _Color:
    def __init__(self):
        self.r = self.g = self.b = self.a = 0.0
class _Scale:
    def __init__(self):
        self.x = self.y = self.z = 0.0
class _Marker:
    SPHERE = 2; ADD = 0; TEXT_VIEW_FACING = 9
    def __init__(self):
        self.header = _Header(); self.ns = ""; self.id = 0; self.type = 0
        self.action = 0; self.pose = _Pose(); self.scale = _Scale()
        self.color = _Color(); self.text = ""
class _MarkerArray:
    def __init__(self):
        self.markers = []

geo = _mod("geometry_msgs"); _mod("geometry_msgs.msg")
sys.modules["geometry_msgs.msg"].PoseArray = _PoseArray
sys.modules["geometry_msgs.msg"].Pose = _Pose
sens = _mod("sensor_msgs"); _mod("sensor_msgs.msg")
sys.modules["sensor_msgs.msg"].Image = _Image
sys.modules["sensor_msgs.msg"].CameraInfo = _CameraInfo
std = _mod("std_msgs"); _mod("std_msgs.msg")
sys.modules["std_msgs.msg"].String = _String
vis = _mod("visualization_msgs"); _mod("visualization_msgs.msg")
sys.modules["visualization_msgs.msg"].Marker = _Marker
sys.modules["visualization_msgs.msg"].MarkerArray = _MarkerArray

tf2 = _mod("tf2_ros")
class _Buffer:
    def __init__(self, *a, **k):
        pass
class _TransformListener:
    def __init__(self, *a, **k):
        pass
tf2.Buffer = _Buffer
tf2.TransformListener = _TransformListener

_mod("cv2")  # never called by the geometry tests

# ---------------------------------------------------------------------------
# 2. Import the REAL node modules from the package source.
# ---------------------------------------------------------------------------
PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "macadamia_sweep", "macadamia_sweep")
sys.path.insert(0, PKG)

import nut_detector as ND
import nut_tracker as NT

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {name}" + (f"  -- {detail}" if detail else ""))

def approx(a, b, tol):
    return abs(a - b) <= tol

# Real OAK-D intrinsics (captured from /oak/rgb/camera_info)
FX, FY, CX, CY = 620.8014, 620.6971, 391.3835, 205.9316

def make_detector(use_depth=False, depth_img=None):
    d = ND.NutDetector.__new__(ND.NutDetector)   # bypass __init__
    d.fx, d.fy, d.cx, d.cy = FX, FY, CX, CY
    d.ground_z = 0.0
    d.min_range, d.max_range = 0.30, 2.50
    d.min_nut_d, d.max_nut_d = 0.015, 0.08
    d.use_depth_gate = use_depth
    d._depth_m = depth_img
    d.depth_floor_tol = 0.10
    d.target_frame = "map"
    return d

def camera_pose_in_map():
    """Compose the real captured transforms:
       map <- base_link  (t=[-0.690,-0.121,0], quat=[0,0,0.480,0.877])
       base_link <- oak_rgb_camera_optical_frame (t=[-0.024,0,0.192],
                                                  quat=[-0.5,0.5,-0.5,0.5])
       Returns (origin[3], R[3x3]) for map <- optical.
    """
    R = ND.quat_to_rotation_matrix
    R_mb = R(0.0, 0.0, 0.480, 0.877)
    t_mb = np.array([-0.690, -0.121, 0.0])
    R_bo = R(-0.5, 0.5, -0.5, 0.5)
    t_bo = np.array([-0.024, 0.0, 0.192])
    R_mo = R_mb @ R_bo
    o = t_mb + R_mb @ t_bo
    return o, R_mo

def project_into_image(P_map, origin, R_mo):
    """Forward pinhole model: map floor point -> pixel (u,v) + optical Z."""
    p_opt = R_mo.T @ (np.asarray(P_map) - origin)   # map -> optical
    X, Y, Z = p_opt
    u = FX * X / Z + CX
    v = FY * Y / Z + CY
    return u, v, Z

# ===========================================================================
print("=== nut_detector: geometry round-trip (real intrinsics + real TF) ===")
origin, R_mo = camera_pose_in_map()
print(f"  camera origin in map = [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]  "
      f"(expect z=0.192)")
check("camera height == 0.192 m", approx(origin[2], 0.192, 1e-3),
      f"z={origin[2]:.4f}")

# Put a nut on the floor ~0.8 m ahead along the camera's forward azimuth.
fwd = R_mo @ np.array([0.0, 0.0, 1.0])
az = math.atan2(fwd[1], fwd[0])
P = np.array([origin[0] + 0.8 * math.cos(az),
              origin[1] + 0.8 * math.sin(az),
              0.0])
u, v, Z = project_into_image(P, origin, R_mo)
print(f"  nut@map=({P[0]:.3f},{P[1]:.3f},0) -> pixel=({u:.1f},{v:.1f}), opticalZ={Z:.3f}")
check("pixel inside 768x432 image", 0 <= u <= 768 and 0 <= v <= 432, f"u={u:.1f} v={v:.1f}")
check("floor point is below horizon (v>cy)", v > CY, f"v={v:.1f} cy={CY:.1f}")

det = make_detector(use_depth=False)
slant = float(np.linalg.norm(P - origin))
radius_px = FX * 0.015 / slant       # a real 3 cm nut at this range
res = det.project_and_gate(u, v, radius_px, origin, R_mo)
check("project_and_gate returns a point", res is not None)
if res is not None:
    err = math.hypot(res[0] - P[0], res[1] - P[1])
    print(f"  recovered=({res[0]:.3f},{res[1]:.3f},{res[2]:.3f})  err={err*1000:.2f} mm")
    check("round-trip recovers nut within 2 mm", err < 0.002, f"err={err*1000:.3f} mm")
    check("recovered z == floor (0)", approx(res[2], 0.0, 1e-6))

# ===========================================================================
print("=== nut_detector: gates ===")
# Oversized blob (e.g. a chair part projected to floor): big pixel radius.
big = det.project_and_gate(u, v, FX * 0.15 / slant, origin, R_mo)
check("physical-size gate rejects 0.30 m blob", big is None)

# Tiny speck below the min diameter.
tiny = det.project_and_gate(u, v, FX * 0.005 / slant, origin, R_mo)
check("physical-size gate rejects 1 cm speck", tiny is None)

# Range gate: a nut 3.0 m ahead (> max_range 2.5).
Pfar = np.array([origin[0] + 3.0 * math.cos(az), origin[1] + 3.0 * math.sin(az), 0.0])
uf, vf, Zf = project_into_image(Pfar, origin, R_mo)
far = det.project_and_gate(uf, vf, FX * 0.015 / float(np.linalg.norm(Pfar - origin)),
                           origin, R_mo)
check("range gate rejects nut at 3.0 m", far is None)

# A ray that points UP (above horizon) must be rejected.
up = det.project_and_gate(CX, 10.0, 10.0, origin, R_mo)   # v<<cy -> upward ray
check("upward ray (sky/wall) rejected", up is None)

# ===========================================================================
print("=== nut_detector: depth on-floor gate ===")
# Build a depth image (metres) that says 'floor' at the nut pixel.
ui, vi = int(round(u)), int(round(v))
depth_floor = np.full((432, 768), np.nan, dtype=np.float32)
depth_floor[vi - 4:vi + 5, ui - 4:ui + 5] = Z          # matches predicted floor
det_df = make_detector(use_depth=True, depth_img=depth_floor)
on_floor = det_df.project_and_gate(u, v, radius_px, origin, R_mo)
check("depth gate ACCEPTS a nut lying on the floor", on_floor is not None)

# Now say the surface is 0.5 m closer than the floor -> standing object.
depth_stand = np.full((432, 768), np.nan, dtype=np.float32)
depth_stand[vi - 4:vi + 5, ui - 4:ui + 5] = Z - 0.5
det_ds = make_detector(use_depth=True, depth_img=depth_stand)
standing = det_ds.project_and_gate(u, v, radius_px, origin, R_mo)
check("depth gate REJECTS a surface standing up (chair)", standing is None)

# Missing depth (textureless card) -> accept (safe asymmetry).
det_nan = make_detector(use_depth=True, depth_img=np.full((432, 768), np.nan, np.float32))
missing = det_nan.project_and_gate(u, v, radius_px, origin, R_mo)
check("depth gate ACCEPTS when depth is unavailable", missing is not None)

# ===========================================================================
print("=== nut_detector: decoders ===")
# depth_to_metres: 16UC1 millimetres, 0 -> NaN.
img = _Image()
img.height, img.width, img.step, img.encoding = 2, 2, 4, "16UC1"
img.data = struct.pack("<4H", 1000, 0, 2000, 500)   # mm
dm = ND.depth_to_metres(img)
check("16UC1 decodes mm->m", dm is not None and approx(dm[0, 0], 1.0, 1e-6)
      and approx(dm[1, 0], 2.0, 1e-6) and approx(dm[1, 1], 0.5, 1e-6),
      f"{None if dm is None else dm.tolist()}")
check("16UC1 maps 0 -> NaN (invalid)", dm is not None and math.isnan(dm[0, 1]))

# image_to_bgr: rgb8 should be flipped to bgr.
rgb = _Image(); rgb.height, rgb.width, rgb.step, rgb.encoding = 1, 1, 3, "rgb8"
rgb.data = bytes([10, 20, 30])
bgr = ND.image_to_bgr(rgb)
check("rgb8 decoded and flipped to BGR", bgr is not None and list(bgr[0, 0]) == [30, 20, 10],
      None if bgr is None else list(bgr[0, 0]))
bg = _Image(); bg.height, bg.width, bg.step, bg.encoding = 1, 1, 3, "bgr8"
bg.data = bytes([10, 20, 30])
bgr2 = ND.image_to_bgr(bg)
check("bgr8 decoded as-is", bgr2 is not None and list(bgr2[0, 0]) == [10, 20, 30])

# ===========================================================================
print("=== nut_tracker: association / confirmation ===")
def make_tracker():
    t = NT.NutTracker.__new__(NT.NutTracker)
    t.nuts = []; t.next_id = 0
    t.merge_radius = 0.15; t.min_hits = 3; t.collection_radius = 0.25
    t.map_frame = "map"; t.robot_frame = "base_link"
    t.marker_ns = "nuts"; t.marker_diameter = 0.08
    return t

trk = make_tracker()
trk.associate(1.00, 1.00)
trk.associate(1.04, 0.98)   # within merge_radius -> same nut
trk.associate(1.02, 1.01)   # same nut again -> hits=3
check("3 nearby detections collapse to ONE nut", len(trk.nuts) == 1, f"n={len(trk.nuts)}")
check("that nut has 3 hits", trk.nuts[0].hits == 3, f"hits={trk.nuts[0].hits}")
trk.associate(2.50, 2.50)   # far -> new nut
check("a far detection starts a new nut", len(trk.nuts) == 2)
conf = trk.confirmed()
check("only the >=3-hit nut is confirmed", len(conf) == 1 and conf[0].id == 0,
      f"confirmed={[n.id for n in conf]}")
avg_ok = approx(trk.nuts[0].x, 1.02, 0.03) and approx(trk.nuts[0].y, 1.00, 0.03)
check("merged position is the running average", avg_ok,
      f"({trk.nuts[0].x:.3f},{trk.nuts[0].y:.3f})")

# ===========================================================================
print("=== nut_tracker: drive-over collection ===")
# Stub clock / logger / publishers and a TF buffer that places the robot.
class _Clk:
    class _N:
        nanoseconds = 123456789
        def to_msg(self):
            return types.SimpleNamespace(sec=0, nanosec=0)
    def now(self):
        return _Clk._N()
class _Log:
    def info(self, *a, **k):
        pass
class _Pub:
    def __init__(self):
        self.last = None
    def publish(self, m):
        self.last = m
class _TF:
    def __init__(self, x, y):
        self.x = x; self.y = y
    def lookup_transform(self, target, source, t):
        ns = types.SimpleNamespace
        return ns(transform=ns(translation=ns(x=self.x, y=self.y, z=0.0)))

trk.get_clock = lambda: _Clk()
trk.get_logger = lambda: _Log()
trk.uncollected_pub = _Pub(); trk.status_pub = _Pub(); trk.marker_pub = _Pub()
# Robot drives onto the confirmed nut #0 (at ~1.02,1.00).
trk.tf_buffer = _TF(1.02, 1.00)
trk.collection_tick()
check("nut under the robot becomes collected", trk.nuts[0].collected is True)
check("far nut stays uncollected", trk.nuts[1].collected is False)
check("/snc_status reports collected 1 / total 1",
      trk.status_pub.last is not None and trk.status_pub.last.data == "Nuts: collected 1 / total 1",
      None if trk.status_pub.last is None else trk.status_pub.last.data)
check("uncollected list is now empty (only confirmed nut was collected)",
      trk.uncollected_pub.last is not None and len(trk.uncollected_pub.last.poses) == 0)

# Collected position must FREEZE: a later stray sighting must not move it.
fx0, fy0 = trk.nuts[0].x, trk.nuts[0].y
trk.associate(1.20, 1.20)   # near-ish the collected nut
check("collected nut position is frozen", approx(trk.nuts[0].x, fx0, 1e-9)
      and approx(trk.nuts[0].y, fy0, 1e-9))

# ===========================================================================
print("=== nut_tracker: marker colours ===")
trk.publish_markers()
ma = trk.marker_pub.last
spheres = [m for m in ma.markers if m.type == _Marker.SPHERE]
collected_m = [m for m in spheres if m.id == 0][0]
check("collected nut sphere is GREEN", collected_m.color.g > collected_m.color.r,
      f"r={collected_m.color.r} g={collected_m.color.g}")
# Promote nut #1 to confirmed + uncollected, re-publish, check it's red.
trk.nuts[1].hits = 3
trk.publish_markers()
ma = trk.marker_pub.last
unc = [m for m in ma.markers if m.type == _Marker.SPHERE and m.id == 1][0]
check("uncollected nut sphere is RED", unc.color.r > unc.color.g,
      f"r={unc.color.r} g={unc.color.g}")
has_label = any(m.type == _Marker.TEXT_VIEW_FACING for m in ma.markers)
check("a TEXT loss-estimate marker is published", has_label)

# ===========================================================================
passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 60)
print(f"SMOKE TEST: {passed}/{total} checks passed")
print("=" * 60)
sys.exit(0 if passed == total else 1)
