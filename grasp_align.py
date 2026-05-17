import json
import math
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import websocket
except Exception:
    websocket = None

try:
    from hp60_sdk_linux import HP60SDKCamera
except Exception:
    try:
        from hp60_sdk import HP60SDKCamera
    except Exception:
        HP60SDKCamera = None


# =========================
# USER SETTINGS
# =========================

PROJECT_DIR = Path(__file__).resolve().parent
HP60_SDK_ROOT = PROJECT_DIR / "EaiCameraSdk_v1.2.28.20241015"

# Camera/depth source
CAMERA_ID = 0
USE_HP60_SDK = True
ALLOW_WEBCAM_FALLBACK = False
HP60_FRAME_WAIT_SECONDS = 8.0

# YOLO model
YOLO_MODEL_PATH = str(PROJECT_DIR / "best3.pt")
#YOLO_MODEL_PATH = str(PROJECT_DIR / "navybest.pt")
YOLO_CONF_THRES = 0.40
YOLO_DEVICE = "cpu"  # "auto", "cpu", or "cuda:0"
YOLO_IMGSZ = 256  ## 256 320  640#######################################
YOLO_USE_HALF_ON_CUDA = True
YOLO_EVERY_N_FRAMES = 5




#USE_EE_HEIGHT_FOR_FORWARD = True
#    use /ee_pose z height when available

#CAMERA_Z_OFFSET_FROM_EE_MM = -30.0
 #   camera is 30 mm below the end-effector/gripper
#
#CUP_TARGET_HEIGHT_FROM_GROUND_MM = 0.0
 #   assume the target/cup reference height is ground height for now

#MIMIC_EE_POSE_Z_MM = 300.0
   # fake end-effector height for testing when USE_MIMIC = True
#The height used in the triangle is:


# Alignment logic
CENTER_TOLERANCE_PX = 15
STABLE_SECONDS = 3.0
CAMERA_TO_GRIPPER_HEIGHT_MM = 30.0
USE_EE_HEIGHT_FOR_FORWARD = True
CAMERA_Z_OFFSET_FROM_EE_MM = -30.0
CUP_TARGET_HEIGHT_FROM_GROUND_MM = 0.0
ALIGN_JOG_HZ = 10.0
FORWARD_PUBLISH_HZ = 1.0
MAX_LOOP_FPS = 0  # set 60.0, 30.0, 20.0, or 0/None to disable
HEADLESS = False #################false/havescreen###############################
END_AFTER_STABLE = True ##################
AVERAGE_FPS = False
AVERAGE_FPS_SECONDS = 10.0

##########################################################

# USE from topic or default templte
USE_EE_POSE = True
# USE mimic ver or use default templte or use from topic, if "USE_EE_POSE = True"
USE_MIMIC =  True
##########################################################



MIMIC_EE_POSE_YAW_DEG = 0 # fake yaw angle
MIMIC_EE_POSE_Z_MM = 300.0 # fake end-effector height from ground/base
EE_YAW_SIGN = 1.0 # controls whether we use the yaw direction normally or flipped. (have to test it)
EE_YAW_OFFSET_DEG = 0.0 #what?
PRINT_JOG_DEBUG = False

# Depth sampling
DEPTH_SAMPLE_RADIUS = 3
DEPTH_MULTI_SAMPLE_RADIUS = 4
DEPTH_BOX_INSET_RATIO = 0.30
MIN_VALID_DEPTH_MM = 150.0
MAX_VALID_DEPTH_MM = 5000.0

# ROS bridge publishing for Windows development.
# If this cannot connect, the camera loop still continues.
ENABLE_ROSBRIDGE_PUBLISH = True
ROSBRIDGE_URL = "ws://10.9.139.118:9090"
ROS_TOPIC = "/goto_position"
ROS_MESSAGE_TYPE = "std_msgs/msg/String"
EE_POSE_TOPIC = "/ee_pose"
TASK_PLANNING_PUBLISH = False ##############################
TASK_STATE_TOPIC = "/task_planner/current_state"
TASK_STATE_MESSAGE_TYPE = "std_msgs/msg/String"
ALIGN_PERPENDICULAR_SUCCESS_STATE = [
    "at(perpendicular)",
]

# Message format:
# Jog command while aligning:
# data = '{"label":"jog","controlMode":"effector","axis":"y","direction":-1,
#          "speed":30,"tcp_x":0,"tcp_y":70.7,"tcp_z":0}'
ALIGN_JOG_TEMPLATE = {
    "label": "jog",
    "controlMode": "effector",
    "axis": "y",
    "direction": 1,
    "speed": 30,
    "tcp_x": 0,
    "tcp_y": 70.7,
    "tcp_z": 0,
}

# Final command after the cup is centered for STABLE_SECONDS:
# data = '{"label":"forward","forward":304.4}'

WINDOW_NAME = "Grasp Alignment"
SHOW_TEXT_ON_WINDOW = True


# =========================
# ROSBRIDGE PUBLISHER
# =========================

class OptionalRosbridgePublisher:
    """
    Best-effort rosbridge publisher for Windows development.

    If rosbridge is offline, this object quietly stays disconnected and the
    camera loop keeps running. That way testing the camera is not disrupted.
    """
    def __init__(self, ws_url, topic, message_type, enabled=True):
        self.ws_url = ws_url
        self.topic = topic
        self.message_type = message_type
        self.enabled = enabled and websocket is not None
        self.ws = None
        self.connected = False
        self.advertised = False
        self.queue = queue.Queue(maxsize=3)

        if not self.enabled:
            print("ROS publish disabled or websocket package not available.")
            return

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

    def _on_open(self, ws):
        self.connected = True
        advertise_msg = {
            "op": "advertise",
            "topic": self.topic,
            "type": self.message_type
        }
        try:
            ws.send(json.dumps(advertise_msg))
            self.advertised = True
            print(f"ROS publish connected: {self.topic}")
        except Exception as e:
            self.connected = False
            self.advertised = False
            print("ROS advertise failed:", e)

    def _on_close(self, ws, *args):
        self.connected = False
        self.advertised = False

    def _on_error(self, ws, error):
        self.connected = False
        self.advertised = False

    def _run(self):
        while self.enabled:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_close=self._on_close,
                    on_error=self._on_error
                )
                self.ws.run_forever()
            except Exception:
                self.connected = False
                self.advertised = False

            # Avoid busy-looping when no rosbridge server is available.
            time.sleep(2.0)

    def publish(self, data):
        if not self.enabled or not self.connected or not self.advertised:
            return

        try:
            self.queue.put_nowait(str(data))
        except queue.Full:
            pass

    def _sender_loop(self):
        while self.enabled:
            try:
                data = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self.connected or not self.advertised or self.ws is None:
                continue

            try:
                msg = {
                    "op": "publish",
                    "topic": self.topic,
                    "msg": {"data": data}
                }
                self.ws.send(json.dumps(msg))
            except Exception:
                self.connected = False
                self.advertised = False


class OptionalRosbridgeEEPoseSubscriber:
    """
    Best-effort /ee_pose subscriber for Windows development.

    Yaw is used for jog axis selection. Z height is used to estimate horizontal
    forward distance when the camera/gripper is higher in the air.
    """
    def __init__(self, ws_url, topic, enabled=True):
        self.ws_url = ws_url
        self.topic = topic
        self.enabled = enabled and websocket is not None
        self.latest_yaw_deg = None
        self.latest_z_mm = None
        self.connected = False

        if not self.enabled:
            print("EE pose subscribe disabled or websocket package not available.")
            return

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _on_open(self, ws):
        self.connected = True
        subscribe_msg = {
            "op": "subscribe",
            "topic": self.topic,
            "type": "geometry_msgs/msg/Pose"
        }
        try:
            ws.send(json.dumps(subscribe_msg))
            print(f"EE pose subscribed: {self.topic}")
        except Exception as e:
            self.connected = False
            print("EE pose subscribe failed:", e)

    def _on_message(self, _ws, message):
        try:
            msg = json.loads(message)
            if msg.get("topic") != self.topic:
                return

            pose_msg = msg.get("msg", {})
            pos = pose_msg.get("position", {})
            ori = pose_msg.get("orientation", {})

            self.latest_z_mm = float(pos.get("z", 0.0)) * 1000.0

            qx = float(ori.get("x", 0.0))
            qy = float(ori.get("y", 0.0))
            qz = float(ori.get("z", 0.0))
            qw = float(ori.get("w", 1.0))

            self.latest_yaw_deg = quaternion_to_yaw_deg(qx, qy, qz, qw)
        except Exception:
            pass

    def _on_close(self, _ws, *args):
        self.connected = False

    def _on_error(self, _ws, _error):
        self.connected = False

    def _run(self):
        while self.enabled:
            try:
                ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_close=self._on_close,
                    on_error=self._on_error
                )
                ws.run_forever()
            except Exception:
                self.connected = False

            # Avoid busy-looping when no rosbridge server is available.
            time.sleep(2.0)


# =========================
# CAMERA HELPERS
# =========================

hp60_camera = None
hp60_camera_failed = False
webcam_cap = None


def get_rgb_and_depth_frames():
    global hp60_camera, hp60_camera_failed, webcam_cap

    if USE_HP60_SDK and not hp60_camera_failed:
        if hp60_camera is None and HP60SDKCamera is not None:
            try:
                sdk_root = HP60_SDK_ROOT
                hp60_camera = HP60SDKCamera(sdk_root=sdk_root, width=640, height=480, fps=10)
                hp60_camera.start()
                print("HP60 SDK source started.")
            except Exception as e:
                hp60_camera_failed = True
                print("HP60 SDK source failed:", e)

        if hp60_camera is not None:
            rgb, depth = hp60_camera.get_latest_frames()

            if hp60_camera.open_failed:
                hp60_camera_failed = True
                print("HP60 SDK source failed:", hp60_camera.last_error)
                return False, None, None

            if rgb is None:
                elapsed = time.time() - hp60_camera.last_frame_time
                if elapsed < HP60_FRAME_WAIT_SECONDS:
                    time.sleep(0.02)
                    return True, np.zeros((480, 640, 3), dtype=np.uint8), None

                hp60_camera_failed = True
                print("HP60 SDK source produced no RGB frames.")
                return False, None, None

            return True, rgb, depth

    if not ALLOW_WEBCAM_FALLBACK:
        return False, None, None

    if webcam_cap is None:
        webcam_cap = cv2.VideoCapture(CAMERA_ID)

    ok, frame = webcam_cap.read()
    if not ok:
        return False, None, None
    return True, frame, None


def normalize_depth_mm(depth_frame, values):
    if values.size == 0:
        return None

    depth_value = float(np.median(values))
    if np.issubdtype(depth_frame.dtype, np.floating) and depth_value < 20.0:
        depth_mm = depth_value * 1000.0
    else:
        depth_mm = depth_value

    if depth_mm < MIN_VALID_DEPTH_MM or depth_mm > MAX_VALID_DEPTH_MM:
        return None
    return depth_mm


def sample_depth_mm(depth_frame, px, py, radius=DEPTH_SAMPLE_RADIUS):
    if depth_frame is None:
        return None

    h, w = depth_frame.shape[:2]
    px = int(round(px))
    py = int(round(py))

    x1 = max(0, px - radius)
    x2 = min(w, px + radius + 1)
    y1 = max(0, py - radius)
    y2 = min(h, py + radius + 1)

    roi = depth_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi = roi.astype(np.float64)
    valid = np.isfinite(roi) & (roi > 0)
    return normalize_depth_mm(depth_frame, roi[valid])


def sample_depth_mm_in_box(depth_frame, x1, y1, x2, y2, inset_ratio=DEPTH_BOX_INSET_RATIO):
    if depth_frame is None:
        return None

    h, w = depth_frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    inset_x = int(box_w * inset_ratio)
    inset_y = int(box_h * inset_ratio)

    sx1 = min(x2 - 1, x1 + inset_x)
    sy1 = min(y2 - 1, y1 + inset_y)
    sx2 = max(sx1 + 1, x2 - inset_x)
    sy2 = max(sy1 + 1, y2 - inset_y)

    roi = depth_frame[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None

    roi = roi.astype(np.float64)
    valid = np.isfinite(roi) & (roi > 0)
    return normalize_depth_mm(depth_frame, roi[valid])


def sample_depth_mm_multi_point(depth_frame, x1, y1, x2, y2):
    """
    Lightweight cup-depth fallback.

    The points stay away from YOLO-box edges/corners so we are less likely to
    sample the wall behind the cup. We use a low percentile of valid patch
    depths so the result prefers the cup surface, but is less noisy than one
    single nearest pixel.
    """
    if depth_frame is None:
        return None

    h, w = depth_frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    # Center-biased points: avoid corners because cup shape is not box-shaped.
    sample_points = [
        (0.50, 0.35),
        (0.42, 0.45),
        (0.58, 0.45),
        (0.50, 0.50),
        (0.42, 0.60),
        (0.58, 0.60),
        (0.50, 0.72),
        (0.50, 0.84),
    ]

    depths = []
    for rx, ry in sample_points:
        px = x1 + rx * box_w
        py = y1 + ry * box_h
        depth_mm = sample_depth_mm(
            depth_frame,
            px,
            py,
            radius=DEPTH_MULTI_SAMPLE_RADIUS
        )
        if depth_mm is not None:
            depths.append(depth_mm)

    if not depths:
        return None

    depths = np.array(depths, dtype=np.float64)
    if depths.size == 1:
        return float(depths[0])

    return float(np.percentile(depths, 25))


def get_mimic_ee_pose_z_mm():
    return float(MIMIC_EE_POSE_Z_MM)


def get_active_ee_z_mm():
    if not USE_EE_POSE:
        return None, "default"
    if USE_MIMIC:
        return get_mimic_ee_pose_z_mm(), "mimic"
    return ee_pose_sub.latest_z_mm, "ee_pose"


def estimate_vertical_height_mm(ee_z_mm=None):
    if USE_EE_HEIGHT_FOR_FORWARD and ee_z_mm is not None:
        camera_z_mm = float(ee_z_mm) + float(CAMERA_Z_OFFSET_FROM_EE_MM)
        vertical_height_mm = camera_z_mm - float(CUP_TARGET_HEIGHT_FROM_GROUND_MM)
        return max(0.0, abs(vertical_height_mm))

    return float(CAMERA_TO_GRIPPER_HEIGHT_MM)


def estimate_forward_distance_mm(depth_distance_mm, ee_z_mm=None):
    if depth_distance_mm is None:
        return None

    c = float(depth_distance_mm)
    b = estimate_vertical_height_mm(ee_z_mm)
    if c <= b:
        return 0.0

    return math.sqrt((c * c) - (b * b))


def quaternion_to_yaw_deg(qx, qy, qz, qw):
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        return 0.0

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return (yaw_deg + 180) % 360.0 # becaseu some how it got offset by 180 degreee, maybe becaseu of the opposite direction, but i just put this + 180 here and it work good


def get_mimic_ee_pose_yaw_deg():
    """
    Fake /ee_pose yaw for testing when the robot/rosbridge server is not open.

    Change MIMIC_EE_POSE_YAW_DEG near the top of the file to test:
        0, 90, 180, 270, etc.
    """
    return float(MIMIC_EE_POSE_YAW_DEG) % 360.0


def correct_ee_yaw_deg(yaw_deg):
    if yaw_deg is None:
        return None
    return ((float(yaw_deg) * EE_YAW_SIGN) + EE_YAW_OFFSET_DEG) % 360.0


def choose_jog_axis_direction(error_distance_px, yaw_deg=None):
    """
    Choose base-frame jog axis from end-effector yaw.

    Fallback behavior matches the old working simulation setup:
        cup right -> axis y, direction -1
        cup left  -> axis y, direction 1
    """
    cup_is_right = error_distance_px > 0 #########################################################################################################

    if not USE_EE_POSE or yaw_deg is None:
        return "y", -1 if cup_is_right else 1

    yaw = correct_ee_yaw_deg(yaw_deg)

    if yaw <= 45.0 or yaw > 315.0:
        axis = "y"
        direction = -1 if cup_is_right else 1
    elif yaw <= 135.0:
        axis = "x"
        direction = 1 if cup_is_right else -1
    elif yaw <= 225.0:
        axis = "y"
        direction = 1 if cup_is_right else -1
    else:
        axis = "x"
        direction = -1 if cup_is_right else 1

    return axis, direction


def build_align_jog_message(error_distance_px, yaw_deg=None):
    command = ALIGN_JOG_TEMPLATE.copy()
    axis, direction = choose_jog_axis_direction(error_distance_px, yaw_deg)
    command["axis"] = axis
    command["direction"] = direction
    return json.dumps(command, separators=(",", ":"))


def build_forward_message(forward_distance_mm):
    command = {
        "label": "forward",
    }
    command["forward"] = round(float(forward_distance_mm), 1)
    return json.dumps(command, separators=(",", ":"))


def get_active_yaw_deg():
    if not USE_EE_POSE:
        return None, "default"
    if USE_MIMIC:
        return get_mimic_ee_pose_yaw_deg(), "mimic"
    return ee_pose_sub.latest_yaw_deg, "ee_pose"


# =========================
# YOLO HELPERS
# =========================

def get_best_cup_detection(results, model_names, conf_thres):
    best_box = None
    best_conf = -1.0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = model_names[cls_id]

            if class_name != "coffee_cup":
                continue
            if conf < conf_thres:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2, conf)

    return best_box


def resolve_yolo_runtime(device_setting):
    if device_setting != "auto":
        use_half = YOLO_USE_HALF_ON_CUDA and str(device_setting).startswith("cuda")
        return device_setting, use_half

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0", YOLO_USE_HALF_ON_CUDA
    except Exception:
        pass

    return "cpu", False


def get_frame_signature(frame):
    """
    Lightweight fingerprint for detecting a new camera image.

    HP60 get_latest_frames() can return the same latest frame multiple times if
    the Python loop is faster than the camera stream. Sampling a small grid lets
    the FPS counter ignore repeated frames and estimate real camera-frame FPS.
    """
    if frame is None:
        return None

    sampled = frame[::16, ::16]
    return sampled.shape, sampled.tobytes()


def get_camera_frame_token(frame):
    """
    Return a token that changes only when a new camera frame is available.

    HP60 provides a timestamp for its latest frame, which is better than image
    hashing because the scene can be static. Webcam fallback treats every
    successful read as a new camera frame.
    """
    if hp60_camera is not None and getattr(hp60_camera, "latest_rgb", None) is not None:
        return ("hp60", float(getattr(hp60_camera, "latest_timestamp", 0.0)))

    if webcam_cap is not None:
        return ("webcam", time.time())

    return ("signature", get_frame_signature(frame))


def update_camera_fps(frame, prev_signature, prev_time, prev_fps):
    signature = get_camera_frame_token(frame)
    if signature is None:
        return prev_signature, prev_time, prev_fps, False

    if prev_signature is not None and signature == prev_signature:
        return prev_signature, prev_time, prev_fps, False

    now = time.time()
    if prev_time is None:
        return signature, now, 0.0, True

    dt = now - prev_time
    instant_fps = 0.0 if dt <= 1e-9 else 1.0 / dt

    return signature, now, instant_fps, True


def update_average_fps_test(new_camera_frame, cup_detected, state):
    if not AVERAGE_FPS:
        return state

    now = time.time()

    if not state["started"]:
        if new_camera_frame and cup_detected:
            state["started"] = True
            state["start_time"] = now
            state["frame_count"] = 0
            print(
                "Average camera FPS test started after first cup detection "
                f"for {AVERAGE_FPS_SECONDS:.1f} seconds."
            )
        return state

    if state["done"]:
        return state

    if new_camera_frame:
        state["frame_count"] += 1

    elapsed = now - state["start_time"]
    if elapsed >= AVERAGE_FPS_SECONDS:
        state["done"] = True
        state["elapsed"] = elapsed
        state["average_fps"] = state["frame_count"] / elapsed if elapsed > 0 else 0.0
        print(
            "Average camera FPS result: "
            f"{state['average_fps']:.2f} FPS "
            f"({state['frame_count']} frames / {elapsed:.2f} s)"
        )

    return state


def get_average_fps_status_text(state):
    if not AVERAGE_FPS:
        return None

    if state["done"]:
        return f"avg_camera_fps={state['average_fps']:.2f} ({state['elapsed']:.1f}s)"

    if state["started"]:
        elapsed = time.time() - state["start_time"]
        return (
            f"avg_fps_test={elapsed:.1f}/{AVERAGE_FPS_SECONDS:.1f}s "
            f"frames={state['frame_count']}"
        )

    return "avg_fps_test=waiting for first cup detection"


def apply_fps_limit(loop_start_time):
    if not MAX_LOOP_FPS or MAX_LOOP_FPS <= 0:
        return

    min_frame_time = 1.0 / float(MAX_LOOP_FPS)
    elapsed = time.time() - loop_start_time
    sleep_time = min_frame_time - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)


def draw_overlay(
    display,
    data,
    cup_box=None,
    cup_center=None,
    image_center_x=None,
    jog_info=None,
    fps_value=None,
    average_fps_text=None
):
    if cup_box is not None:
        x1, y1, x2, y2, conf = cup_box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            display,
            f"cup {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    if image_center_x is not None:
        cv2.line(
            display,
            (int(image_center_x), 0),
            (int(image_center_x), display.shape[0] - 1),
            (255, 255, 255),
            1
        )

    if cup_center is not None:
        cv2.circle(display, cup_center, 6, (0, 0, 255), -1)

    if SHOW_TEXT_ON_WINDOW:
        error_px, centered, stable, depth_mm, forward_mm = data
        lines = [
            f"error_distance_px={error_px:.1f}",
            f"centered={bool(centered)} stable={bool(stable)}",
            f"depth={depth_mm:.1f} mm",
            f"forward={forward_mm:.1f} mm",
        ]
        if fps_value is not None:
            lines.append(f"camera_fps={fps_value:.1f}")
        if average_fps_text is not None:
            lines.append(average_fps_text)
        if jog_info is not None:
            yaw_raw = jog_info.get("yaw_raw")
            yaw_used = jog_info.get("yaw_used")
            axis = jog_info.get("axis", "?")
            direction = jog_info.get("direction", "?")
            source = jog_info.get("source", "?")
            ee_z_mm = jog_info.get("ee_z_mm")
            ee_z_source = jog_info.get("ee_z_source", "?")
            vertical_height_mm = jog_info.get("vertical_height_mm")
            if yaw_raw is None:
                lines.append(f"jog axis={axis} dir={direction} yaw=None src={source}")
            else:
                lines.append(
                    f"jog axis={axis} dir={direction} "
                    f"yaw={yaw_raw:.1f}->{yaw_used:.1f} src={source}"
                )
            if ee_z_mm is None:
                lines.append(
                    f"height={vertical_height_mm:.1f}mm src=fixed"
                )
            else:
                lines.append(
                    f"ee_z={ee_z_mm:.1f}mm height={vertical_height_mm:.1f}mm src={ee_z_source}"
                )

        for i, line in enumerate(lines):
            cv2.putText(
                display,
                line,
                (20, 35 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )


# =========================
# MAIN PROGRAM
# =========================

yolo_runtime_device, yolo_runtime_half = resolve_yolo_runtime(YOLO_DEVICE)
model = YOLO(YOLO_MODEL_PATH)
try:
    model.fuse()
except Exception:
    pass
print("Loaded YOLO model:", YOLO_MODEL_PATH)
print(f"YOLO runtime: device={yolo_runtime_device}, imgsz={YOLO_IMGSZ}, half={yolo_runtime_half}")
if HEADLESS:
    print("Headless mode enabled. Press Ctrl+C to quit.")
else:
    print("Press q to quit.")

publisher = OptionalRosbridgePublisher(
    ROSBRIDGE_URL,
    ROS_TOPIC,
    ROS_MESSAGE_TYPE,
    enabled=ENABLE_ROSBRIDGE_PUBLISH
)
task_state_publisher = OptionalRosbridgePublisher(
    ROSBRIDGE_URL,
    TASK_STATE_TOPIC,
    TASK_STATE_MESSAGE_TYPE,
    enabled=ENABLE_ROSBRIDGE_PUBLISH and TASK_PLANNING_PUBLISH
)
ee_pose_sub = OptionalRosbridgeEEPoseSubscriber(
    ROSBRIDGE_URL,
    EE_POSE_TOPIC,
    enabled=ENABLE_ROSBRIDGE_PUBLISH and USE_EE_POSE and not USE_MIMIC
)

centered_start_time = None
last_print_time = 0.0
last_jog_publish_time = 0.0
last_forward_publish_time = 0.0
last_camera_fps_time = None
last_camera_frame_signature = None
fps_value = 0.0
average_fps_state = {
    "started": False,
    "done": False,
    "start_time": None,
    "frame_count": 0,
    "elapsed": 0.0,
    "average_fps": 0.0,
}
yolo_frame_counter = 0
cached_best_cup = None
forward_published_for_current_lock = False

if not HEADLESS:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

try:
    while True:
        loop_start_time = time.time()
    
        ok, frame, depth_frame = get_rgb_and_depth_frames()
        if not ok:
            print("Failed to read frame.")
            break

        (
            last_camera_frame_signature,
            last_camera_fps_time,
            fps_value,
            new_camera_frame
        ) = update_camera_fps(
            frame,
            last_camera_frame_signature,
            last_camera_fps_time,
            fps_value
        )
        display = frame.copy()
        h, w = frame.shape[:2]
        image_center_x = w / 2.0
    
        if yolo_frame_counter % YOLO_EVERY_N_FRAMES == 0:
            results = model(
                frame,
                verbose=False,
                device=yolo_runtime_device,
                imgsz=YOLO_IMGSZ,
                half=yolo_runtime_half
            )
            cached_best_cup = get_best_cup_detection(results, model.names, YOLO_CONF_THRES)
    
        best_cup = cached_best_cup
        yolo_frame_counter += 1
        average_fps_state = update_average_fps_test(
            new_camera_frame,
            best_cup is not None,
            average_fps_state
        )
        average_fps_text = get_average_fps_status_text(average_fps_state)
    
        if best_cup is not None:
            x1, y1, x2, y2, conf = best_cup
            cup_center_x = (x1 + x2) / 2.0
            cup_center_y = (y1 + y2) / 2.0
            cup_center = (int(round(cup_center_x)), int(round(cup_center_y)))
    
            error_distance_px = cup_center_x - image_center_x
            centered = abs(error_distance_px) <= CENTER_TOLERANCE_PX
    
            if centered:
                if centered_start_time is None:
                    centered_start_time = time.time()
                    forward_published_for_current_lock = False
                stable = (time.time() - centered_start_time) >= STABLE_SECONDS
            else:
                centered_start_time = None
                stable = False
                forward_published_for_current_lock = False
    
            depth_distance_mm = None
            if centered:
                depth_distance_mm = sample_depth_mm_in_box(depth_frame, x1, y1, x2, y2)
                if depth_distance_mm is None:
                    depth_distance_mm = sample_depth_mm_multi_point(depth_frame, x1, y1, x2, y2)
                if depth_distance_mm is None:
                    depth_distance_mm = sample_depth_mm(depth_frame, cup_center_x, cup_center_y)
    
            yaw_deg, yaw_source = get_active_yaw_deg()
            ee_z_mm, ee_z_source = get_active_ee_z_mm()
            vertical_height_mm = estimate_vertical_height_mm(ee_z_mm)
            forward_distance_mm = estimate_forward_distance_mm(depth_distance_mm, ee_z_mm)
    
            depth_out = 0.0 if depth_distance_mm is None else float(depth_distance_mm)
            forward_out = 0.0 if forward_distance_mm is None else float(forward_distance_mm)
    
            data = [
                float(error_distance_px),
                1.0 if centered else 0.0,
                1.0 if stable else 0.0,
                depth_out,
                forward_out
            ]
    
            now = time.time()
            yaw_used = correct_ee_yaw_deg(yaw_deg)
            jog_axis, jog_direction = choose_jog_axis_direction(error_distance_px, yaw_deg)
            jog_info = {
                "yaw_raw": yaw_deg,
                "yaw_used": yaw_used,
                "axis": jog_axis,
                "direction": jog_direction,
                "source": yaw_source,
                "ee_z_mm": ee_z_mm,
                "ee_z_source": ee_z_source,
                "vertical_height_mm": vertical_height_mm,
            }
    
            if not centered and now - last_jog_publish_time >= (1.0 / ALIGN_JOG_HZ):
                align_jog_msg = build_align_jog_message(error_distance_px, yaw_deg)
                publisher.publish(align_jog_msg)
                if PRINT_JOG_DEBUG:
                    if yaw_deg is None:
                        print(
                            "jog debug: "
                            f"error_px={error_distance_px:.1f}, yaw=None, "
                            f"axis={jog_axis}, direction={jog_direction}, source={yaw_source}, "
                            f"ee_z={ee_z_mm}, vertical_height={vertical_height_mm:.1f}"
                        )
                    else:
                        print(
                            "jog debug: "
                            f"error_px={error_distance_px:.1f}, "
                            f"yaw_raw={yaw_deg:.1f}, yaw_used={yaw_used:.1f}, "
                            f"axis={jog_axis}, direction={jog_direction}, source={yaw_source}, "
                            f"ee_z={ee_z_mm}, vertical_height={vertical_height_mm:.1f}"
                        )
                last_jog_publish_time = now
    
            if (
                stable
                and forward_distance_mm is not None
                and not forward_published_for_current_lock
                and now - last_forward_publish_time >= (1.0 / FORWARD_PUBLISH_HZ)
            ):
                forward_msg = build_forward_message(forward_distance_mm)
                publisher.publish(forward_msg)
                print("published /goto_position:", forward_msg)
                if TASK_PLANNING_PUBLISH:
                    task_state_msg = json.dumps(ALIGN_PERPENDICULAR_SUCCESS_STATE)
                    task_state_publisher.publish(task_state_msg)
                    print(f"published {TASK_STATE_TOPIC}:", ALIGN_PERPENDICULAR_SUCCESS_STATE)
                last_forward_publish_time = now
                forward_published_for_current_lock = True
                if END_AFTER_STABLE and not (AVERAGE_FPS and not average_fps_state["done"]):
                    print("Stable alignment complete. Exiting.")
                    break
    
            if now - last_print_time >= 0.2:
                print(
                    "data = "
                    f"[{data[0]:.1f}, {data[1]:.0f}, {data[2]:.0f}, "
                    f"{data[3]:.1f}, {data[4]:.1f}]"
                )
                last_print_time = now
    
            draw_overlay(
                display,
                data,
                cup_box=best_cup,
                cup_center=cup_center,
                image_center_x=image_center_x,
                jog_info=jog_info,
                fps_value=fps_value,
                average_fps_text=average_fps_text
            )
        else:
            centered_start_time = None
            forward_published_for_current_lock = False
            cv2.line(
                display,
                (int(image_center_x), 0),
                (int(image_center_x), h - 1),
                (255, 255, 255),
                1
            )
            if SHOW_TEXT_ON_WINDOW:
                cv2.putText(
                    display,
                    "No cup detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    display,
                    f"camera_fps={fps_value:.1f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                if average_fps_text is not None:
                    cv2.putText(
                        display,
                        average_fps_text,
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
    
        if not HEADLESS:
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        apply_fps_limit(loop_start_time)
except KeyboardInterrupt:
    print("Stopped by user.")

if webcam_cap is not None:
    webcam_cap.release()
if hp60_camera is not None:
    hp60_camera.stop()
if not HEADLESS:
    cv2.destroyAllWindows()
