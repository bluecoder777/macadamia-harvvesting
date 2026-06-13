#!/usr/bin/env python3
"""Nut detector — PERCEPTION layer.

Detects macadamia "nuts" (malina-card coloured discs, ~3 cm) on the floor
from the OAK-D RGB image and publishes their position in the `map` frame.

Why this design (decided from the robot's real setup, captured live):
  * The OAK-D on this ROSbot 3 Pro is mounted HORIZONTAL at z=0.192 m
    (TF base_link -> oak_rgb_camera_optical_frame = [-0.024, 0, 0.192],
    optical-only rotation, i.e. NO down-tilt). Vertical FOV ~38 deg, so the
    floor first enters the frame ~0.55 m ahead. Nuts directly under the robot
    are invisible -- which is fine, because we LOCALISE each nut while it is
    still ahead and remember it (see nut_tracker.py).
  * Depth (/oak/stereo/*) is ALIGNED to the RGB optical frame (identical K,
    same frame_id). But OAK-D stereo holes out on textureless flat cardboard,
    so instead of sampling depth we use GROUND-PLANE RAY INTERSECTION: shoot
    the pixel ray through the pinhole model, transform it into `map` with TF,
    and intersect the floor plane z = ground_z. Robust for flat discs on a
    flat indoor floor, and needs only RGB + camera_info + TF.
  * Detect on the RECTIFIED image /oak/rgb/image_rect: it matches the K in
    camera_info with zero distortion, so the pinhole math is exact.
  * Raspberry Pi 5 (4 cores) -> classic HSV + contour CV, no GPU/YOLO needed.

This node is deliberately stateless per-frame. All persistence, de-duplication
and the collected/uncollected bookkeeping live in nut_tracker.py (the world
model). That separation is the three-layer split: perception here, world model
+ mission logic there, reactive control in simple_row_follower.py.

Pipeline (per processed frame):
    RGB(rect) -> HSV -> colour mask (two hue bands for red/raspberry wrap)
              -> morphology -> contours -> area + circularity gate
              -> minEnclosingCircle centroid (u,v)
              -> ground-plane ray intersection in `map`
              -> range gate -> PoseArray (frame_id = map)

Publishes:
    /nuts/detections   geometry_msgs/PoseArray   raw per-frame detections (map)
    /nuts/debug_image  sensor_msgs/Image (bgr8)  overlay for HSV tuning

Run:
    ros2 run macadamia_sweep nut_detector
    # tune the colour live while watching the debug image in rqt_image_view:
    ros2 run rqt_image_view rqt_image_view /nuts/debug_image
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose

import tf2_ros


def image_to_bgr(msg: Image) -> Optional[np.ndarray]:
    """Convert a sensor_msgs/Image to an OpenCV BGR uint8 array WITHOUT
    cv_bridge (one less thing to apt-install on the robot).

    Honours msg.step (row stride / padding) and the common OAK-D encodings.
    Returns None for encodings we don't handle.
    """
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("bgr8", "rgb8"):
        rows = buf.reshape(h, msg.step)[:, : w * 3]
        img = rows.reshape(h, w, 3)
        if enc == "rgb8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)
    if enc in ("bgra8", "rgba8"):
        rows = buf.reshape(h, msg.step)[:, : w * 4]
        img = rows.reshape(h, w, 4)[:, :, :3]
        if enc == "rgba8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)
    return None


def depth_to_metres(msg: Image) -> Optional[np.ndarray]:
    """Decode an aligned depth image to a float32 array in METRES, with invalid
    pixels set to NaN. Handles the two OAK-D depth encodings (16UC1 mm, 32FC1 m).
    """
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, msg.step // 2)[:, :w]
        out = arr.astype(np.float32) / 1000.0
        out[arr == 0] = np.nan
    elif enc == "32fc1":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, msg.step // 4)[:, :w]
        out = np.array(arr, dtype=np.float32, copy=True)
        out[~np.isfinite(out)] = np.nan
        out[out <= 0.0] = np.nan
    else:
        return None
    out[out > 10.0] = np.nan  # nothing useful past 10 m indoors
    return out


def quat_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix from a (x, y, z, w) quaternion."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class NutDetector(Node):
    def __init__(self):
        super().__init__("nut_detector")

        # ---- Topics / frames ----
        self.declare_parameter("rgb_topic", "/oak/rgb/image_rect")
        self.declare_parameter("camera_info_topic", "/oak/rgb/camera_info")
        # Fixed frame nuts are reported in. `map` exists (slam_toolbox). If you
        # run without SLAM, set this to `odom`.
        self.declare_parameter("target_frame", "map")
        # Height of the floor plane in target_frame. base_link sits at z=0 in
        # map on this robot, so the floor is z=0.
        self.declare_parameter("ground_z", 0.0)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.ground_z = float(self.get_parameter("ground_z").value)

        # ---- Colour gate (HSV, OpenCV ranges: H 0-179, S/V 0-255) ----
        # "Malina" = raspberry: a saturated pink/red that wraps the hue circle,
        # so we use TWO bands (low reds 0..h1, high reds/magenta h2..179) and
        # OR them. These are the numbers to tune against /nuts/debug_image.
        self.declare_parameter("h_lo1", 0)
        self.declare_parameter("h_hi1", 10)
        self.declare_parameter("h_lo2", 160)
        self.declare_parameter("h_hi2", 179)
        self.declare_parameter("s_min", 110)
        self.declare_parameter("s_max", 255)
        self.declare_parameter("v_min", 60)
        self.declare_parameter("v_max", 255)

        # ---- Detection mode ----
        # "background": nuts may be ANY colour; instead of matching a nut hue
        #   we SUBTRACT the known floor colour (green astroturf) and keep the
        #   round, floor-sized blobs of whatever is left. Robust when nut
        #   colours vary. (A nut that is itself green will blend in -- avoid
        #   green/lime cards.)
        # "color": the original single-hue gate above (all nuts one colour).
        self.declare_parameter("detect_mode", "background")
        # Floor colour to subtract in background mode (defaults tuned for green
        # astroturf; widen floor_s_min / floor_v_min to swallow shadowed turf).
        self.declare_parameter("floor_h_lo", 30)
        self.declare_parameter("floor_h_hi", 95)
        self.declare_parameter("floor_s_min", 25)
        self.declare_parameter("floor_v_min", 20)
        self.declare_parameter("floor_v_max", 255)
        self.detect_mode = str(self.get_parameter("detect_mode").value).lower()

        # ---- Shape / size gate ----
        self.declare_parameter("min_area_px", 12.0)
        self.declare_parameter("max_area_px", 1500.0)
        # Solidity = contour area / convex-hull area ("is it a FILLED blob?").
        # This is FORESHORTENING-INVARIANT: the camera is horizontal at 0.19 m
        # so a floor disc is viewed at a shallow angle and projects to an
        # ELLIPSE (often very thin), not a circle -- circularity rejected those.
        # A disc at any angle is a filled ellipse (solidity ~0.9); ragged or
        # partial clutter scores lower.
        self.declare_parameter("min_solidity", 0.80)
        # Range-AWARE physical-size gate (the strongest clutter rejector).
        # After the floor projection we know the range, so we check the blob's
        # REAL diameter (m), not just pixels. A nut is ~3 cm; the cap is 5 cm so
        # a 6 cm noodle BASE (which sits on the floor and otherwise passes every
        # gate) is rejected.
        self.declare_parameter("min_nut_diameter_m", 0.015)
        self.declare_parameter("max_nut_diameter_m", 0.05)

        # ---- Depth on-floor gate (rejects things standing UP off the floor:
        # chairs, bags, cupboards) ----
        # Uses the aligned depth image. A nut lies flat, so the measured depth
        # at its pixel matches the floor-plane-predicted distance. Furniture
        # surfaces sit CLOSER than the floor at that pixel -> rejected.
        self.declare_parameter("use_depth_gate", True)
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw")
        # Reject if the measured surface is closer than the predicted floor by
        # more than this (m). Generous, to absorb stereo noise.
        self.declare_parameter("depth_floor_tolerance", 0.10)
        # Ignore everything ABOVE this fraction of image height (the horizon
        # and walls/noodles above it). Camera is horizontal so the horizon is
        # at v=cy~206 of 432 ~ 0.48; 0.45 trims just above it.
        self.declare_parameter("roi_top_fraction", 0.45)
        # Morphology kernel (px). SMALL (3) preserves the thin, foreshortened
        # nut ellipses the low horizontal camera sees. Raise only for a noisy
        # floor (e.g. turf) -- but a big kernel erodes shallow-angle nuts away.
        self.declare_parameter("morph_px", 3)

        # ---- Range gate (m, horizontal distance camera->nut in target_frame) ----
        self.declare_parameter("min_range", 0.30)
        self.declare_parameter("max_range", 2.50)

        # ---- Throttle: process every Nth frame (camera ~14 Hz) ----
        self.declare_parameter("process_every_n", 2)

        self.declare_parameter("publish_debug", True)

        self.min_area = float(self.get_parameter("min_area_px").value)
        self.max_area = float(self.get_parameter("max_area_px").value)
        self.min_solidity = float(self.get_parameter("min_solidity").value)
        self.min_nut_d = float(self.get_parameter("min_nut_diameter_m").value)
        self.max_nut_d = float(self.get_parameter("max_nut_diameter_m").value)
        self.use_depth_gate = bool(self.get_parameter("use_depth_gate").value)
        self.depth_topic = self.get_parameter("depth_topic").value
        self.depth_floor_tol = float(self.get_parameter("depth_floor_tolerance").value)
        self.roi_top_fraction = float(self.get_parameter("roi_top_fraction").value)
        self.morph_px = max(1, int(self.get_parameter("morph_px").value))
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        # ---- Camera intrinsics (filled from camera_info; fall back to the
        # values captured live from this exact robot so the node works even if
        # camera_info is briefly missing) ----
        self.fx = 620.80
        self.fy = 620.70
        self.cx = 391.38
        self.cy = 205.93
        self.have_info = False

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Pub / Sub ----
        self.det_pub = self.create_publisher(PoseArray, "/nuts/detections", 10)
        self.debug_pub = (
            self.create_publisher(Image, "/nuts/debug_image", 2)
            if self.publish_debug
            else None
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self.info_callback, 10
        )
        self.create_subscription(Image, self.rgb_topic, self.image_callback, 10)

        # Latest aligned depth (metres, NaN where invalid). None until first msg.
        self._depth_m: Optional[np.ndarray] = None
        if self.use_depth_gate:
            self.create_subscription(
                Image, self.depth_topic, self.depth_callback, 10
            )

        self.frame_count = 0
        self._debug_tint: Optional[np.ndarray] = None
        self._debug_tint_label = ""

        if self.detect_mode == "background":
            mode_str = (
                f"mode=background (subtract floor "
                f"H[{self.get_parameter('floor_h_lo').value}-"
                f"{self.get_parameter('floor_h_hi').value}])"
            )
        else:
            mode_str = (
                f"mode=color (H[{self.get_parameter('h_lo1').value}-"
                f"{self.get_parameter('h_hi1').value}]+"
                f"[{self.get_parameter('h_lo2').value}-"
                f"{self.get_parameter('h_hi2').value}])"
            )
        depth_str = (
            f"depth-gate ON ({self.depth_topic}, tol {self.depth_floor_tol:.2f}m)"
            if self.use_depth_gate else "depth-gate OFF"
        )
        self.get_logger().info(
            f"nut_detector up. rgb={self.rgb_topic} -> {self.target_frame}. "
            f"{mode_str}. {depth_str}. Tune against /nuts/debug_image."
        )

    # -----------------------------
    # Callbacks
    # -----------------------------

    def info_callback(self, msg: CameraInfo):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        if not self.have_info:
            self.have_info = True
            self.get_logger().info(
                f"camera_info locked: fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f} ({msg.width}x{msg.height})"
            )

    def depth_callback(self, msg: Image):
        depth = depth_to_metres(msg)
        if depth is None:
            self.get_logger().warn(
                f"Unhandled depth encoding '{msg.encoding}'; disabling depth gate.",
                throttle_duration_sec=10.0,
            )
            self.use_depth_gate = False
            return
        self._depth_m = depth

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        bgr = image_to_bgr(msg)
        if bgr is None:
            self.get_logger().warn(
                f"Unhandled image encoding '{msg.encoding}'", throttle_duration_sec=5.0
            )
            return

        mask = self.colour_mask(bgr)
        candidates = self.find_candidates(mask)   # (u, v, radius, shape_status)

        # Project each shape-passing blob onto the floor in target_frame. The
        # camera->map transform is the same for every blob in this frame, so
        # look it up ONCE here rather than per blob.
        source_frame = msg.header.frame_id or "oak_rgb_camera_optical_frame"
        cam = self._lookup_camera_pose(source_frame, msg.header.stamp)
        poses: List[Pose] = []
        # (u, v, radius, draw_status) -- "ok" green, "gate" orange, "shape" yellow.
        draw: List[Tuple[int, int, int, str]] = []
        n_shape_ok = sum(1 for c in candidates if c[3] == "ok")

        if cam is None:
            if n_shape_ok:
                self.get_logger().warn(
                    f"Saw {n_shape_ok} nut blob(s) but TF "
                    f"{self.target_frame}<-{source_frame} was unavailable.",
                    throttle_duration_sec=5.0,
                )
            draw = [(u, v, r, "shape" if st != "ok" else "gate")
                    for (u, v, r, st) in candidates]
        else:
            origin, R = cam
            for (u, v, radius, st) in candidates:
                if st != "ok":
                    draw.append((u, v, radius, "shape"))
                    continue
                pt = self.project_and_gate(u, v, radius, origin, R)
                if pt is None:
                    draw.append((u, v, radius, "gate"))
                    continue
                p = Pose()
                p.position.x, p.position.y, p.position.z = pt
                p.orientation.w = 1.0
                poses.append(p)
                draw.append((u, v, radius, "ok"))

        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.target_frame
        out.poses = poses
        self.det_pub.publish(out)

        if self.debug_pub is not None:
            self.publish_debug_image(bgr, draw, msg.header.stamp)

    # -----------------------------
    # Vision
    # -----------------------------

    def colour_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Return a binary FOREGROUND mask (candidate nut pixels).

        Also stashes self._debug_tint / self._debug_tint_label so the debug
        overlay can show what's being selected (nut colour) or removed (floor).
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        roi_top = int(self.roi_top_fraction * bgr.shape[0])

        if self.detect_mode == "background":
            # Subtract the floor (green astroturf); keep everything else.
            floor = cv2.inRange(
                hsv,
                (int(self.get_parameter("floor_h_lo").value),
                 int(self.get_parameter("floor_s_min").value),
                 int(self.get_parameter("floor_v_min").value)),
                (int(self.get_parameter("floor_h_hi").value),
                 255,
                 int(self.get_parameter("floor_v_max").value)),
            )
            mask = cv2.bitwise_not(floor)
            self._debug_tint = floor
            self._debug_tint_label = "floor removed"
        else:
            # Single-hue gate: nuts share one colour (two bands for red wrap).
            s_min = int(self.get_parameter("s_min").value)
            s_max = int(self.get_parameter("s_max").value)
            v_min = int(self.get_parameter("v_min").value)
            v_max = int(self.get_parameter("v_max").value)
            lo1 = (int(self.get_parameter("h_lo1").value), s_min, v_min)
            hi1 = (int(self.get_parameter("h_hi1").value), s_max, v_max)
            lo2 = (int(self.get_parameter("h_lo2").value), s_min, v_min)
            hi2 = (int(self.get_parameter("h_hi2").value), s_max, v_max)
            mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
            self._debug_tint = mask
            self._debug_tint_label = "nut colour"

        # Kill the horizon and everything above it (walls, noodles, ceiling).
        if roi_top > 0:
            mask[:roi_top, :] = 0

        # Gentle morphology: a SMALL kernel removes speckle without eroding the
        # thin, foreshortened nut ellipses the low horizontal camera sees.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_px, self.morph_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def find_candidates(self, mask: np.ndarray) -> List[Tuple[int, int, int, str]]:
        """Return (u, v, radius_px, status) for every blob in the mask.

        status is "ok" if it passes the area + solidity shape test, else the
        reason it failed ("small" / "big" / "notsolid"). Reporting the failures
        lets the debug image distinguish "in the mask but wrong shape" (drawn
        yellow) from "not in the mask at all" (no circle).

        Centre + radius come from a fitted ELLIPSE: the major axis of a flat
        disc's projection stays ~ its true diameter even when foreshortened, so
        the downstream physical-size gate works at any viewing angle."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        out: List[Tuple[int, int, int, str]] = []
        for c in contours:
            if len(c) >= 5:
                (ex, ey), (ax1, ax2), _ = cv2.fitEllipse(c)
                u, v, radius = int(round(ex)), int(round(ey)), int(round(max(ax1, ax2) / 2.0))
            else:
                (ex, ey), rr = cv2.minEnclosingCircle(c)
                u, v, radius = int(round(ex)), int(round(ey)), int(round(rr))
            area = cv2.contourArea(c)
            if area < self.min_area:
                out.append((u, v, radius, "small")); continue
            if area > self.max_area:
                out.append((u, v, radius, "big")); continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < self.min_solidity:
                out.append((u, v, radius, "notsolid")); continue
            out.append((u, v, radius, "ok"))
        return out

    # -----------------------------
    # Geometry: pixel -> floor point in target_frame
    # -----------------------------

    def _lookup_camera_pose(self, source_frame: str, stamp):
        """Return (origin[3], R[3x3]) of the camera in target_frame, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, source_frame, Time.from_msg(stamp),
                timeout=Duration(seconds=0.1),
            )
        except Exception:
            # Fall back to the latest available transform.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame, source_frame, Time()
                )
            except Exception:
                return None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rotation_matrix(q.x, q.y, q.z, q.w)
        origin = np.array([t.x, t.y, t.z], dtype=float)
        return origin, R

    def project_and_gate(
        self, u: int, v: int, radius_px: float, origin: np.ndarray, R: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Intersect the camera ray through pixel (u,v) with the floor plane
        z = ground_z, then apply the range, physical-size and depth-on-floor
        gates. origin/R are the camera pose in target_frame. Returns (x,y,z)."""
        # Ray direction in the optical frame (z forward, x right, y down). Its
        # z-component is 1, so the scalar `s` below is also the forward optical
        # depth to the floor point -- which is exactly what a depth image
        # measures (used by the depth gate).
        d_opt = np.array(
            [(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0], dtype=float
        )
        d_map = R @ d_opt
        if d_map[2] >= -1e-6:          # ray must head DOWN toward the floor
            return None
        s = (self.ground_z - origin[2]) / d_map[2]
        if s <= 0:
            return None
        point = origin + s * d_map

        # Range gate (horizontal distance camera->nut).
        rng = math.hypot(point[0] - origin[0], point[1] - origin[1])
        if rng < self.min_range or rng > self.max_range:
            return None

        # Range-aware physical-size gate: real diameter = 2*radius_px * R / fx,
        # using slant range as the pixel->metre scale.
        slant = math.hypot(rng, origin[2] - self.ground_z)
        diameter_m = 2.0 * radius_px * slant / self.fx
        if diameter_m < self.min_nut_d or diameter_m > self.max_nut_d:
            return None

        # Depth on-floor gate: reject surfaces standing up in front of the floor.
        if self.use_depth_gate and not self._depth_on_floor(u, v, s, radius_px):
            return None

        return (float(point[0]), float(point[1]), float(point[2]))

    def _depth_on_floor(self, u: int, v: int, s_pred: float, radius_px: float) -> bool:
        """True if the measured depth at the blob is consistent with it lying
        on the floor (or depth is unavailable -> can't judge, so accept).

        s_pred is the forward distance to the floor at this pixel. A surface
        measured clearly CLOSER than that is standing up off the floor."""
        depth = self._depth_m
        if depth is None:
            return True
        u, v = int(round(u)), int(round(v))   # robust to float centroids
        h, w = depth.shape
        if not (0 <= u < w and 0 <= v < h):
            return True
        r = max(2, int(radius_px // 2))
        patch = depth[max(0, v - r):min(h, v + r + 1),
                      max(0, u - r):min(w, u + r + 1)]
        valid = patch[np.isfinite(patch)]
        if valid.size < 3:
            return True   # textureless (e.g. the card itself) -> can't judge
        measured = float(np.median(valid))
        return (s_pred - measured) <= self.depth_floor_tol

    # -----------------------------
    # Debug overlay
    # -----------------------------

    def publish_debug_image(self, bgr, draw, stamp):
        vis = bgr.copy()
        # Tint the selected/removed pixels so tuning is obvious: in background
        # mode this is the FLOOR being subtracted (tune until all floor is
        # tinted); in colour mode it's the selected nut hue. Magenta is chosen
        # so it stays visible over a green OR blue floor.
        tint = self._debug_tint
        if tint is not None:
            vis[tint > 0] = (0.5 * vis[tint > 0] + np.array([160, 0, 160])).astype(np.uint8)
        # Horizon / ROI line.
        roi_top = int(self.roi_top_fraction * bgr.shape[0])
        cv2.line(vis, (0, roi_top), (vis.shape[1], roi_top), (255, 255, 0), 1)
        # Status colours: green = accepted nut, orange = passed shape but a
        # gate rejected it, yellow = found in mask but wrong shape/size.
        colours = {"ok": (0, 255, 0), "gate": (0, 165, 255), "shape": (0, 255, 255)}
        n = {"ok": 0, "gate": 0, "shape": 0}
        for (u, v, radius, st) in draw:
            n[st] = n.get(st, 0) + 1
            colour = colours.get(st, (0, 0, 255))
            cv2.circle(vis, (u, v), max(radius, 3), colour, 2)
            cv2.circle(vis, (u, v), 2, colour, -1)
        cv2.putText(
            vis,
            f"{self._debug_tint_label} | "
            f"green(ok):{n['ok']} orange(gate):{n['gate']} yellow(shape):{n['shape']}",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

        out = Image()
        out.header.stamp = stamp
        out.header.frame_id = "oak_rgb_camera_optical_frame"
        out.height, out.width = vis.shape[0], vis.shape[1]
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = vis.shape[1] * 3
        out.data = vis.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = NutDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
