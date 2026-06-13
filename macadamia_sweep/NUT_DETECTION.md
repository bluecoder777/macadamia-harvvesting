# Nut detection & collection tracking

Two nodes, added to the `macadamia_sweep` package:

| Node | Layer | Role |
|------|-------|------|
| `nut_detector` | Perception | RGB → HSV colour mask → circular blobs → ground-plane projection → `/nuts/detections` (PoseArray in `map`). Stateless per frame. |
| `nut_tracker` | World model / mission | De-dups detections into unique nuts, marks them **collected** when the robot drives over them, publishes RViz spheres + a saved list of uncollected nuts. |

This is the **three-layer** split: perception (`nut_detector`) → world model + mission state (`nut_tracker`) → reactive control (`simple_row_follower`).

## Why these design choices (from the robot's real setup)

- **Camera is horizontal at 0.192 m, no tilt** → floor visible from ~0.55 m ahead, blind cone underneath. We localise each nut while it's ahead and remember it, so the "drive-over = collected" check works without seeing under the robot.
- **Detect on `/oak/rgb/image_rect`** (rectified) → matches `camera_info` `K` with zero distortion, so the pinhole projection is exact.
- **Ground-plane ray intersection, not depth.** OAK-D stereo holes out on flat textureless cardboard; intersecting the pixel ray with the floor (`z=0` in `map`, via TF) is more robust and needs only RGB + `camera_info` + TF.
- **Classic HSV+contour CV** (no YOLO) → runs fine on the Pi 5; deterministic and tunable.

## Build & run

```bash
cd ~/your_ros2_ws            # the workspace this package lives in
colcon build --packages-select macadamia_sweep
source install/setup.bash

# perception + tracker together:
ros2 launch macadamia_sweep nut_detection.launch.py

# in RViz: add a MarkerArray display on /nuts/markers, Fixed Frame = map
#   RED sphere   = uncollected nut
#   GREEN sphere = collected nut (robot drove within collection_radius)
```

Dependencies: `python3-opencv`, `python3-numpy` (no `cv_bridge` needed — images are decoded directly). If missing:
```bash
sudo apt install python3-opencv python3-numpy
```

## Two detection modes

`detect_mode` parameter:

- **`background`** (default) — nuts can be **any colour**; the detector subtracts the known **floor** colour (green astroturf) and keeps round, floor-sized blobs of whatever remains. Use this when nut colours vary. A green/lime nut would blend into the turf — avoid those.
- **`color`** — all nuts share **one** colour; matches a single HSV hue band (two bands for red wrap). Set `detect_mode:=color`.

In both modes the colour-independent **shape gate** (size + **solidity**) does the real discrimination, so it rejects ragged/partial clutter regardless of nut colour. Solidity (filled-ness) is used instead of circularity because the low horizontal camera sees floor discs as foreshortened **ellipses**, not circles — solidity is invariant to that.

### Rejecting clutter (chairs, bags, cupboards)
Five colour-independent gates stack, in order: horizon cut → pixel area → **solidity** → on-floor projection + range → **physical size (m)** → **depth on-floor**. The last two are the heavy lifters for furniture:
- **Physical-size**: after projection the range is known, so the blob's real diameter must be ~1.5–8 cm. A round chair wheel rarely is.
- **Depth on-floor**: the aligned depth must agree the blob *lies on the floor*. A surface standing up (measured closer than the floor plane) is rejected — this targets furniture specifically, regardless of its colour. Textureless nuts that return no depth are accepted (the gate only rejects on positive evidence of standing-up geometry).

Watch it work on `/nuts/debug_image`: rejected blobs draw **orange**, accepted **green** — drive past your chairs and they should stay orange. Residual risk: a small, round, ~3 cm object *lying on the floor* (bottle cap, coin) can still pass — only a spatial sweep-area gate or a learned classifier removes those.

## Calibrate (do this once)

```bash
ros2 launch macadamia_sweep nut_detection.launch.py
ros2 run rqt_image_view rqt_image_view /nuts/debug_image
```
On the debug image: **magenta-tinted pixels = floor being removed** (in background mode), cyan line = horizon/ROI cutoff (above it is ignored). Blob circles: **green** = accepted nut (published), **orange** = passed shape but a gate (size/range/depth) rejected it, **yellow** = found in the mask but wrong shape/size, **no circle** = not in the mask at all. The label counts each. So `green` high = good; `yellow` high = relax `min_solidity` / `min_area_px`; `orange` high = relax a gate; nothing circled = fix the floor mask.

> Note: the camera is horizontal at 0.19 m, so floor discs are foreshortened into ellipses, not circles. The shape gate therefore uses **solidity** (filled-ness, invariant to viewing angle), not circularity. If distant nuts come up yellow, lower `min_solidity` (e.g. 0.7).

Background mode — tune until **all the astroturf is tinted** but the nuts are NOT:
```bash
ros2 param set /nut_detector floor_h_lo 30     # widen the green hue range
ros2 param set /nut_detector floor_h_hi 95
ros2 param set /nut_detector floor_s_min 20    # lower to swallow shadowed/pale turf
ros2 param set /nut_detector floor_v_min 15
```
If turf speckle still sneaks through as tiny blobs, raise `min_area_px`. When happy, copy the values into `nut_detection.launch.py` (or a params yaml).

### Get me a frame to pre-tune the HSV for you
You recorded `nut_sample`. Extract one RGB frame and send it:
```bash
# play the bag and save one frame
ros2 bag play nut_sample &
ros2 run image_view image_saver --ros-args -r image:=/oak/rgb/image_rect \
    -p filename_format:='nut_%04d.jpg'   # Ctrl-C after a couple of frames
# (image_view not installed? quick alternative:)
python3 - <<'PY'
import rclpy, numpy as np, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
rclpy.init(); n=Node("grab")
def cb(m):
    a=np.frombuffer(m.data,np.uint8).reshape(m.height,m.step)[:, :m.width*3].reshape(m.height,m.width,3)
    if m.encoding=='rgb8': a=a[:,:,::-1]
    cv2.imwrite('nut_frame.png', a); print('wrote nut_frame.png'); rclpy.shutdown()
n.create_subscription(Image,'/oak/rgb/image_rect',cb,1); rclpy.spin(n)
PY
```

## Key parameters

`nut_detector`:
- `detect_mode` — `background` (subtract floor, any nut colour) or `color` (single nut hue).
- `floor_h_lo/floor_h_hi`, `floor_s_min`, `floor_v_min/floor_v_max` — floor (astroturf) colour to subtract in **background** mode.
- `h_lo1/h_hi1`, `h_lo2/h_hi2`, `s_min/s_max`, `v_min/v_max` — HSV nut gate in **color** mode (two hue bands for red wrap).
- `min_area_px` / `max_area_px`, `min_solidity` — shape gate (colour-independent; the real nut/not-nut discriminator). Solidity = filled-ness, robust to the foreshortened-ellipse look of floor discs.
- `min_nut_diameter_m` / `max_nut_diameter_m` (0.015 / 0.05) — range-aware **physical-size** gate: real diameter must be nut-sized. The 5 cm cap rejects 6 cm noodle bases.
- `morph_px` (3) — morphology kernel size. Keep small: a large kernel erodes the thin foreshortened nut ellipses away. Raise only for a noisy floor.
- `use_depth_gate` (true), `depth_topic`, `depth_floor_tolerance` (0.10 m) — **depth on-floor** gate: rejects surfaces standing up off the floor (chairs/bags/cupboards). Set `use_depth_gate:=false` to A/B it.
- `roi_top_fraction` (0.45) — ignore image above the horizon.
- `min_range` / `max_range` (0.30 / 2.50 m) — reject too-near/too-far projections.
- `process_every_n` (2) — throttle vs the ~14 Hz camera.

`nut_tracker`:
- `merge_radius` (0.15 m) — two detections within this are the same nut.
- `min_hits` (3) — sightings before a nut is confirmed/shown.
- `collection_radius` (0.25 m) — robot within this of a nut → collected. Set to your sweeper/footprint half-width.
- `marker_diameter` (0.08 m) — sphere size in RViz.
- `save_path` (`~/nut_locations.csv`) — written on shutdown.

## Outputs (for the later "collect all uncollected in one sweep" phase)

- `/nuts/uncollected` — `geometry_msgs/PoseArray`, **latched** (transient-local). A picker node started later still receives the full list. Plan a route over these (a TSP / coverage problem).
- `~/nut_locations.csv` — `id, x_map, y_map, collected, hits`, written on Ctrl-C.

## Verifying collection (pickup) works

"Picked up" = the tracker flipped a nut from uncollected→collected because `base_link` passed within `collection_radius`. Verify it in this order.

### RViz
```bash
rviz2 -d $(ros2 pkg prefix macadamia_sweep)/share/macadamia_sweep/rviz/nuts.rviz
# or from the source tree: rviz2 -d macadamia_sweep/rviz/nuts.rviz
```
The config sets **Fixed Frame = map** and shows: `/nuts/markers` (spheres), `/nuts/debug_image`, `/nuts/uncollected` (red arrows), `/map`, `/scan`, TF. If you build the displays by hand instead, the only non-obvious settings are: Map and `/nuts/uncollected` need **Durability = Transient Local**, and `/scan` needs **Reliability = Best Effort**.

What you should see: a nut appears as a **RED sphere**, and when the robot drives over it the sphere turns **GREEN** and the floating `collected X/Y` label increments.

### Controlled test (no camera needed)
Isolates the tracker/collection logic from HSV tuning:
```bash
ros2 run macadamia_sweep nut_tracker                 # if not already running
python3 tools/inject_fake_nuts.py --ahead 0.6        # drops a fake nut 0.6 m ahead
```
Watch a RED sphere appear ~0.6 m in front of the robot. Then teleop forward over it:
```bash
ros2 topic echo /snc_status      # should go "...collected 1 / total 1"
```
The sphere should turn GREEN as `base_link` crosses within `collection_radius`.

### Cross-checks (terminal)
```bash
ros2 topic echo /nuts/detections   # raw per-frame detections (map frame)
ros2 topic echo /snc_status        # "Nuts: collected X / total Y"
ros2 topic echo /nuts/uncollected  # shrinks by one each time a nut is collected
cat ~/nut_locations.csv            # on shutdown: id,x,y,collected,hits
```

### If a nut never turns green
- `collection_radius` too small for how close you actually drive → raise it (`ros2 param set /nut_tracker collection_radius 0.30`).
- The nut never **confirmed** (needs `min_hits=3` sightings) → it won't show or collect; lower `min_hits` or dwell on it longer.
- TF `map→base_link` not available to the tracker → no robot pose, no collection. Check `ros2 run tf2_ros tf2_echo map base_link`.
- Detections arriving in a frame other than `map` → tracker ignores them (it warns).

## Note on running with the row follower
`simple_row_follower` and Nav2 both drive `/cmd_vel_nav` — run only one of them at a time. The nut nodes are **read-only** w.r.t. motion (they never publish velocity), so they're safe to run alongside either.
