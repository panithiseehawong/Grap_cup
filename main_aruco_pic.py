import cv2
import numpy as np
from ultralytics import YOLO
import json
import websocket
import math
import time
import threading
import queue
from pathlib import Path

try:
    from hp60_sdk_linux import HP60SDKCamera
except Exception:
    try:
        from hp60_sdk import HP60SDKCamera
    except Exception:
        HP60SDKCamera = None


# This program does 4 main jobs:
# 1) Open camera and detect the cup with YOLO
# 2) Detect the ArUco marker to get real-world reference
# 3) Convert cup position from camera frame to robot base frame
# 4) Read /ee_pose from ROS and publish the final cup position to ROS



# =========================
# USER SETTINGS
# =========================

# chosee the camera
PROJECT_DIR = Path(__file__).resolve().parent
HP60_SDK_ROOT = PROJECT_DIR / "EaiCameraSdk_v1.2.28.20241015"
CAMERA_ID = 0
USE_DEPTH_ASSIST = True
DEPTH_FRAME_SOURCE = "hp60_sdk"
ALLOW_WEBCAM_FALLBACK = False
DEPTH_SAMPLE_RADIUS = 2
DEPTH_BOX_INSET_RATIO = 0.15
MIN_VALID_DEPTH_MM = 150
MAX_VALID_DEPTH_MM = 5000
DEPTH_VALUE_IS_RADIAL_DISTANCE = True
HP60_FRAME_WAIT_SECONDS = 8.0

# YOLO model
YOLO_MODEL_PATH = str(PROJECT_DIR / "best3.pt")
YOLO_CONF_THRES = 0.40
YOLO_DEVICE = "cpu" # "auto", "cpu", or "cuda:0"
YOLO_IMGSZ = 256
#YOLO_EVERY_N_FRAMES = 3

# ArUco marker settings
ARUCO_MARKER_ID = 0
ARUCO_MARKER_LENGTH = 0.130
ARUCO_DETECT_SCALE = 1.5
#ARUCO_EVERY_N_FRAMES = 2

# ArUco dictionary
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Camera calibration file
CALIB_FILE = str(PROJECT_DIR / "charo_deck" / "aruco_1.npz")

WINDOW_NAME = "Cup Position Snapshot"

# Cup size in meters
CUP_DIAMETER = 0.08
CUP_HEIGHT = 0.12
CUP_HEIGHT_AXIS_SIGN = 1.0

# Grasp height ratio
# 0.5 = middle of cup
GRASP_HEIGHT_RATIO = 0.0

# Camera-to-gripper local offsets in mm.
# Local X = down, Local Y = left/right, Local Z = horizontal forward.
X_OFFSET_CAMERA_MM = 0.0
Y_OFFSET_CAMERA_MM = 0.0
Z_OFFSET_CAMERA_MM = 0.0

# Contour tuning
MIN_CONTOUR_AREA = 800
MIN_CONTOUR_HEIGHT = 40
MIN_CONTOUR_WIDTH = 15
LOWER_REGION_RATIO = 0.75
SEARCH_START_RATIO = 0.45
FALLBACK_Y_RATIO = 0.85

# Canny
CANNY_LOW = 50
CANNY_HIGH = 150

# Short-term memory
MAX_LOST_FRAMES = 8

# Display
HEADLESS = False ####fasle/showscreen####################################3
SHOW_DEBUG_WINDOWS = False
SHOW_TEXT_ON_MAIN_WINDOW = True
FPS_SMOOTHING = 0.90
PRINT_POSITION_LOG = False
PUBLISH_EVERY_N_FRAMES = 10
SNAPSHOT_WARMUP_FRAMES = 12
SNAPSHOT_CAPTURE_FRAMES = 5
SNAPSHOT_FRAME_DELAY_SEC = 0.03
SNAPSHOT_CAPTURE_INTERVAL_SEC = 0.5
SAVE_DEBUG_IMAGE = False
DEBUG_IMAGE_PATH = str(PROJECT_DIR / "main_aruco_pic_result.png")
TIME_DURATION = True


####
USE_MIMIC_EE_POSE = True
####
EE_POSE_WAIT_SECONDS = 2.0
PUBLISH_REPEAT_COUNT = 3
PUBLISH_REPEAT_INTERVAL_SEC = 0.2
PUBLISH_CONNECT_TIMEOUT_SEC = 2.0

# Rosbridge
WS_URL = "ws://10.9.139.118:9090"
GET_POSE_EVERY_N_FRAMES = 10
TASK_PLANNING_PUBLISH = False ###################################
TASK_INITIAL_TOPIC = "/task_planner/initial_state"
TASK_GOAL_TOPIC = "/task_planner/goal_state"
TASK_STATE_TOPIC = "/task_planner/current_state"
TASK_STATE_MESSAGE_TYPE = "std_msgs/msg/String"
TASK_INITIAL_STATE = [
    "at(search)",
    "see(cup)",
    "hand_empty",
]
TASK_GOAL_STATE = [
    "at(perpendicular)",
]
SEARCH_CUP_SUCCESS_STATE = [
    "cup_centered",
    "cup_reachable",
]


# =========================
# LOAD CALIBRATION
# =========================

calib = np.load(CALIB_FILE)
camera_matrix = calib["camera_matrix"]
dist_coeffs = calib["dist_coeffs"]

print("Loaded calibration.")
print("camera_matrix =")
print(camera_matrix)
print("dist_coeffs =")
print(dist_coeffs)


# =========================
# CREATE ARUCO DETECTOR
# =========================

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 53
detector_params.adaptiveThreshWinSizeStep = 4
print("Prepared ArUco dictionary and detector parameters.")

aruco_detector = None
if hasattr(cv2.aruco, "ArucoDetector"):
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)


def detect_aruco_markers(gray):
    if aruco_detector is not None:
        return aruco_detector.detectMarkers(gray)

    if hasattr(cv2.aruco, "detectMarkers"):
        return cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detector_params
        )

    raise AttributeError("OpenCV aruco marker detection API is not available.")


# =========================
# LOAD YOLO
# =========================

model = YOLO(YOLO_MODEL_PATH)
try:
    model.fuse()
except Exception:
    pass
print("Loaded YOLO model:", YOLO_MODEL_PATH)
print(f"YOLO runtime: device={YOLO_DEVICE}, imgsz={YOLO_IMGSZ}")


# =========================
# SHORT-TERM MEMORY STATE
# =========================

last_good_point = None
lost_count = 0


# =========================
# HELPER FUNCTIONS
# =========================

import json
import websocket
import time

hp60_camera = None
hp60_camera_failed = False
webcam_cap = None


def get_rgb_and_depth_frames():
    """
    Get camera frames for the vision loop.

    Preferred path:
        HP60C SDK -> RGB frame + depth frame

    Optional development fallback:
        webcam only, if ALLOW_WEBCAM_FALLBACK is True

    For real robot use, ALLOW_WEBCAM_FALLBACK should stay False so the program
    does not accidentally use a normal webcam when the depth camera is missing.
    """
    global hp60_camera, hp60_camera_failed, webcam_cap

    if DEPTH_FRAME_SOURCE == "hp60_sdk" and not hp60_camera_failed:
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
            rgb_frame, depth_frame = hp60_camera.get_latest_frames()
            if rgb_frame is not None:
                return True, rgb_frame, depth_frame

            if hp60_camera.open_failed:
                hp60_camera_failed = True
                print("HP60 SDK source failed:", hp60_camera.last_error)
                hp60_camera.stop()
                hp60_camera = None
            else:
                elapsed = time.time() - hp60_camera.last_frame_time
                if elapsed < HP60_FRAME_WAIT_SECONDS:
                    wait_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        wait_frame,
                        "Waiting for HP60 SDK frames...",
                        (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
                    time.sleep(0.03)
                    return True, wait_frame, depth_frame

                hp60_camera_failed = True
                print("HP60 SDK source produced no frames.")
                hp60_camera.stop()
                hp60_camera = None

    if not ALLOW_WEBCAM_FALLBACK:
        return False, None, None

    if webcam_cap is None:
        webcam_cap = cv2.VideoCapture(CAMERA_ID)
        if not webcam_cap.isOpened():
            return False, None, None

    ok, rgb_frame = webcam_cap.read()
    return ok, rgb_frame, None


def capture_snapshot_frames():
    """
    Warm up the camera, then collect several real frame pairs.

    For picture mode we do not need a live loop. We just wait a little so the
    HP60 stream settles, then process several final RGB/depth snapshots and
    choose the best one.
    """
    captured_frames = []
    valid_count = 0
    capture_count = 0
    deadline = time.time() + HP60_FRAME_WAIT_SECONDS + 2.0
    capture_start_time = time.time()
    last_capture_time = None

    while time.time() < deadline:
        ok, rgb_frame, depth_frame = get_rgb_and_depth_frames()
        if not ok:
            time.sleep(SNAPSHOT_FRAME_DELAY_SEC)
            continue

        # For HP60 snapshot mode, do not accept the synthetic placeholder frame
        # that says "Waiting for HP60 SDK frames...". Only accept real SDK data.
        if DEPTH_FRAME_SOURCE == "hp60_sdk":
            if hp60_camera is None:
                time.sleep(SNAPSHOT_FRAME_DELAY_SEC)
                continue
            real_rgb, real_depth = hp60_camera.get_latest_frames()
            if real_rgb is None:
                time.sleep(SNAPSHOT_FRAME_DELAY_SEC)
                continue
            rgb_frame = real_rgb
            depth_frame = real_depth

        if rgb_frame is not None and np.any(rgb_frame):
            valid_count += 1

            if valid_count > SNAPSHOT_WARMUP_FRAMES:
                now = time.time()
                if last_capture_time is None or (now - last_capture_time) >= SNAPSHOT_CAPTURE_INTERVAL_SEC:
                    captured_frames.append((
                        rgb_frame.copy(),
                        None if depth_frame is None else depth_frame.copy()
                    ))
                    capture_count += 1
                    last_capture_time = now
                    elapsed = now - capture_start_time
                    print(
                        f"Captured snapshot frame {capture_count}/{SNAPSHOT_CAPTURE_FRAMES} "
                        f"at t={elapsed:.2f}s"
                    )

                    if not HEADLESS:
                        preview = rgb_frame.copy()
                        cv2.putText(
                            preview,
                            f"Captured snapshot {capture_count}/{SNAPSHOT_CAPTURE_FRAMES}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )
                        cv2.imshow(WINDOW_NAME, preview)
                        cv2.waitKey(1)

                    if len(captured_frames) >= SNAPSHOT_CAPTURE_FRAMES:
                        break

        time.sleep(SNAPSHOT_FRAME_DELAY_SEC)

    if not captured_frames:
        return False, []

    return True, captured_frames


def sample_depth_mm(depth_frame, px, py, radius=DEPTH_SAMPLE_RADIUS):
    if depth_frame is None:
        return None

    h, w = depth_frame.shape[:2]
    if px < 0 or px >= w or py < 0 or py >= h:
        return None

    x1 = max(0, px - radius)
    y1 = max(0, py - radius)
    x2 = min(w, px + radius + 1)
    y2 = min(h, py + radius + 1)

    patch = depth_frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    patch = patch.astype(np.float64)
    valid = np.isfinite(patch) & (patch > 0)
    values = patch[valid]
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
    values = roi[valid]
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


def pixel_depth_to_camera_3d(u, v, depth_mm, camera_matrix):
    """
    Convert one depth pixel into OpenCV camera coordinates.

    HP60 depth behaves like camera-to-point distance. When that is true, the
    forward Z component is a little smaller than the raw depth for off-center
    pixels, because the raw depth follows the camera ray.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy
    depth_m = depth_mm / 1000.0

    if DEPTH_VALUE_IS_RADIAL_DISTANCE:
        ray = np.array([x, y, 1.0], dtype=np.float64)
        ray /= np.linalg.norm(ray)
        return ray * depth_m

    Z = depth_m
    X = x * Z
    Y = y * Z

    return np.array([X, Y, Z], dtype=np.float64)


def estimate_grasp_point_from_depth(depth_frame, px, py, camera_matrix, cup_box=None):
    """
    Build the cup 3D point from real HP60C depth.

    First try median depth inside the YOLO cup box.
    If that fails, try a tiny patch around the selected red point.
    """
    depth_mm = None
    if cup_box is not None:
        x1, y1, x2, y2 = cup_box
        depth_mm = sample_depth_mm_in_box(depth_frame, x1, y1, x2, y2)

    if depth_mm is None:
        depth_mm = sample_depth_mm(depth_frame, px, py)

    if depth_mm is None:
        return None, None

    grasp_cam = pixel_depth_to_camera_3d(px, py, depth_mm, camera_matrix)
    if not np.all(np.isfinite(grasp_cam)):
        return None, None

    return grasp_cam, depth_mm


def camera_to_board_point(P_cam, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    P_cam = np.array(P_cam, dtype=np.float64).reshape(3)
    P_board = R.T @ (P_cam - t)
    return P_board

class CupPositionPublisher:

    """
    This class sends the final cup position to ROS through rosbridge.

    It publishes to:
        /cup_position_base_mm

    Message type:
        std_msgs/msg/Float64MultiArray

    The data format is:
        [x_mm, y_mm, z_mm]

    Why this class uses threads:
    - The camera loop must stay fast
    - Sending data through WebSocket can be slow/blocking
    - So we put sending work in a background thread
    - This prevents the whole vision program from freezing
    """
    def __init__(self, ws_url):

        # ws_url = rosbridge WebSocket address, for example:
    # ws://10.9.139.118:9090

    # self.ws = actual WebSocket object
    # self.connected = True when connected to rosbridge
    # self.advertised = True when topic has been announced to rosbridge

    # queue is used to store outgoing messages safely
    # maxsize=5 prevents unlimited memory growth if sending is slower than camera loop

        
        self.ws_url = ws_url
        self.ws = None
        self.connected = False
        self.advertised = False

        self.queue = queue.Queue(maxsize=5)  # prevent memory leak

        # WebSocket thread
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

        # Sender thread (IMPORTANT)
        self.sender_thread = threading.Thread(target=self._sender_loop)
        self.sender_thread.daemon = True
        self.sender_thread.start()

    def _on_open(self, ws):
        # This runs when WebSocket connection is successful.

        # We first "advertise" the topic to rosbridge.
        # That tells ROS:
        # "I want to publish this topic and this is its message type."
        print("Publisher connected")
        self.connected = True

        advertise_msg = {
            "op": "advertise",
            "topic": "/cup_position_base_mm",
            "type": "std_msgs/msg/Float64MultiArray"
        }

        try:
            ws.send(json.dumps(advertise_msg))
            self.advertised = True
            print("Advertised /cup_position_base_mm")
        except Exception as e:
            print("Advertise failed:", e)

    def _on_close(self, ws, *args):
        print("Publisher disconnected")
        self.connected = False
        self.advertised = False

    def _on_error(self, ws, error):
        print("Publisher error:", error)

    def close(self):
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def _run(self):

        # This creates and runs the WebSocket client in a background thread.
        # It keeps the ROS connection separate from the camera loop.
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_close=self._on_close,
            on_error=self._on_error
        )
        self.ws.run_forever()

        # 🔥 NON-BLOCKING publish
    def publish(self, x_mm, y_mm, z_mm):

        # This function does NOT send immediately.
        # It only puts data into a queue.
    
        # This is important:
        # if we send directly here, the camera loop may freeze.
        # So this function is intentionally non-blocking.
        if not self.connected or not self.advertised:
            return

        try:
            self.queue.put_nowait((x_mm, y_mm, z_mm))
        except queue.Full:
            pass  # drop old data (important for real-time)

    
        # 🔥 BACKGROUND sender (no freeze)
    def _sender_loop(self):

        # This runs forever in the background.
        # It waits for new position data from the queue.
        # When data is available, it sends it to rosbridge.
    
        # So the flow is:
        # camera loop -> queue -> sender thread -> rosbridge -> ROS topic
        while True:
            try:
                x_mm, y_mm, z_mm = self.queue.get()

                msg = {
                    "op": "publish",
                    "topic": "/cup_position_base_mm",
                    "msg": {
                        "data": [float(x_mm), float(y_mm), float(z_mm)]
                    }
                }

                if self.connected and self.advertised:
                    self.ws.send(json.dumps(msg))

            except Exception as e:
                print("Send error:", e)


class TaskPlannerStatePublisher:
    def __init__(self, ws_url, topic=TASK_STATE_TOPIC, message_type=TASK_STATE_MESSAGE_TYPE):
        self.ws_url = ws_url
        self.topic = topic
        self.message_type = message_type
        self.ws = None
        self.connected = False
        self.advertised = False
        self.queue = queue.Queue(maxsize=3)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

    def _on_open(self, ws):
        self.connected = True
        advertise_msg = {
            "op": "advertise",
            "topic": self.topic,
            "type": self.message_type,
        }
        try:
            ws.send(json.dumps(advertise_msg))
            self.advertised = True
            print(f"Advertised {self.topic}")
        except Exception as e:
            self.connected = False
            self.advertised = False
            print("Task state advertise failed:", e)

    def _on_close(self, ws, *args):
        self.connected = False
        self.advertised = False

    def _on_error(self, ws, error):
        self.connected = False
        self.advertised = False
        print("Task state publisher error:", error)

    def close(self):
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def _run(self):
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_close=self._on_close,
            on_error=self._on_error,
        )
        self.ws.run_forever()

    def publish(self, state_predicates):
        if not self.connected or not self.advertised:
            return

        try:
            self.queue.put_nowait(list(state_predicates))
        except queue.Full:
            pass

    def _sender_loop(self):
        while True:
            try:
                state_predicates = self.queue.get()
                msg = {
                    "op": "publish",
                    "topic": self.topic,
                    "msg": {"data": json.dumps(state_predicates)},
                }
                if self.connected and self.advertised:
                    self.ws.send(json.dumps(msg))
            except Exception as e:
                print("Task state send error:", e)


def print_duration_report(timing_marks):
    if not TIME_DURATION:
        return

    print("Time duration report:")
    previous_name = None
    previous_time = None
    for name, mark_time in timing_marks:
        if previous_time is not None:
            print(f"  {previous_name} -> {name}: {mark_time - previous_time:.2f}s")
        previous_name = name
        previous_time = mark_time

    if timing_marks:
        total = timing_marks[-1][1] - timing_marks[0][1]
        print(f"  total: {total:.2f}s")


def wait_for_one_ee_pose(ws_url, timeout_sec=EE_POSE_WAIT_SECONDS):
    if USE_MIMIC_EE_POSE:
        return get_one_end_effector_pose_mimic(), "mimic"

    ee_sub = EEPoseSubscriber(ws_url)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if ee_sub.latest_pose is not None:
            pose = ee_sub.latest_pose
            ee_sub.close()
            return pose, "ee_pose"
        time.sleep(0.02)

    ee_sub.close()
    return None, "timeout"


def publish_result_once(base_x, base_y, base_z):
    publisher = CupPositionPublisher(WS_URL)
    deadline = time.time() + PUBLISH_CONNECT_TIMEOUT_SEC

    while time.time() < deadline:
        if publisher.connected and publisher.advertised:
            break
        time.sleep(0.02)

    if not publisher.connected or not publisher.advertised:
        print("Publisher could not connect before timeout.")
        publisher.close()
        return False

    for _ in range(PUBLISH_REPEAT_COUNT):
        publisher.publish(base_x, base_y, base_z)
        time.sleep(PUBLISH_REPEAT_INTERVAL_SEC)

    publisher.close()
    return True


def publish_task_predicates_once(topic, state_predicates):
    publisher = TaskPlannerStatePublisher(WS_URL, topic=topic)
    deadline = time.time() + PUBLISH_CONNECT_TIMEOUT_SEC

    while time.time() < deadline:
        if publisher.connected and publisher.advertised:
            break
        time.sleep(0.02)

    if not publisher.connected or not publisher.advertised:
        print("Task state publisher could not connect before timeout.")
        publisher.close()
        return False

    for _ in range(PUBLISH_REPEAT_COUNT):
        publisher.publish(state_predicates)
        time.sleep(PUBLISH_REPEAT_INTERVAL_SEC)

    publisher.close()
    return True


class EEPoseSubscriber:

    """
    This class continuously listens to /ee_pose from ROS through rosbridge.

    Topic:
        /ee_pose

    Message type:
        geometry_msgs/msg/Pose

    It stores the latest robot gripper pose in:
        self.latest_pose

    Why this class is needed:
    - The robot end-effector pose changes over time
    - The vision code needs the latest gripper pose to convert cup position into base frame
    - We receive it continuously in background instead of requesting it inside the camera loop
    """
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.latest_pose = None
        self.ws = None

        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _on_open(self, ws):

        # This runs after WebSocket connects.
        # It subscribes to /ee_pose so rosbridge starts sending pose messages to this code.
        print("EE Pose connected")

        ws.send(json.dumps({
            "op": "subscribe",
            "topic": "/ee_pose",
            "type": "geometry_msgs/msg/Pose"
        }))

    def _on_message(self, ws, message):


         # This runs every time a new /ee_pose message is received.

        # It:
        # 1) reads position from ROS message
        # 2) converts meters -> millimeters
        # 3) reads quaternion orientation
        # 4) converts quaternion -> Euler angles (for display/debug only)
        # 5) stores everything in self.latest_pose
    
        # self.latest_pose is the newest robot pose available to the main vision loop
        try:
            msg = json.loads(message)

            if msg.get("topic") == "/ee_pose":
                pose_msg = msg.get("msg", {})

                pos = pose_msg.get("position", {})
                ori = pose_msg.get("orientation", {})

                x_mm = float(pos.get("x", 0.0)) * 1000.0
                y_mm = float(pos.get("y", 0.0)) * 1000.0
                z_mm = float(pos.get("z", 0.0)) * 1000.0

                qx = float(ori.get("x", 0.0))
                qy = float(ori.get("y", 0.0))
                qz = float(ori.get("z", 0.0))
                qw = float(ori.get("w", 1.0))

                roll, pitch, yaw = quaternion_to_euler_deg(qx, qy, qz, qw)

                self.latest_pose = {
                    "x": x_mm,
                    "y": y_mm,
                    "z": z_mm,
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "qw": qw,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw
                }

        except Exception as e:
            print("EE pose error:", e)

    def _on_error(self, ws, error):
        print("EE Pose error:", error)

    def close(self):
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def _run(self):

        # This starts the subscriber WebSocket client in a background thread.
        # So the camera loop does not wait/block for ROS messages.
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error
        )
        self.ws.run_forever()



def quaternion_to_euler_deg(x, y, z, w):

    """
    Convert quaternion orientation into roll, pitch, yaw in degrees.

    Input:
        quaternion (x, y, z, w)

    Output:
        roll, pitch, yaw in degrees

    Why this is used:
    - ROS /ee_pose gives orientation as quaternion
    - Quaternion is good for math, but hard for humans to read
    - Roll/pitch/yaw is easier to print on screen for debugging

    Important:
    - For actual coordinate transformation, the code uses quaternion -> rotation matrix
    - This Euler output is mainly for display/debug
    """

    
    # roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw)
    )


def quaternion_to_rotation_matrix(qx, qy, qz, qw):

    """
    Convert quaternion into a 3x3 rotation matrix.

    Why this is important:
    - The cup is first found in the camera/gripper local frame
    - To convert that local position into the robot base frame,
      we must rotate the local vector using the robot orientation

    Output:
        3x3 numpy rotation matrix
    """

    
    # normalize for safety
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        return np.eye(3, dtype=np.float64)

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)

    return R


# mimic output from /ee_pose topic, where my friend isn't be here to public it for me
# the value is the default state of pos of the gripper compared to the robot arm
def get_one_end_effector_pose_mimic(): #
    """
    Fake /ee_pose for testing when real ROS pose is not available.

    It returns the same format as the real ee_pose subscriber:
        {
            "x": ...,
            "y": ...,
            "z": ...,
            "qx": ...,
            "qy": ...,
            "qz": ...,
            "qw": ...,
            "roll": ...,
            "pitch": ...,
            "yaw": ...
        }

    Why this exists:
    - Sometimes the robot or friend laptop is not available
    - We still want to test the vision + transform pipeline
    """

    # ---- GIVEN VALUES (from your friend system) ----
    pos = {
        "x": 0.59306,
        "y": 2.03352600998418e-17,
        "z": 0.28787
    }

    ori = {
        "x": -2.1648901405887335e-17,
        "y": 0.7071067811865475,
        "z": 2.164890140588733e-17,
        "w": 0.7071067811865476
        
    }

    # ---- convert position (m → mm) ----
    x_mm = pos["x"] * 1000.0
    y_mm = pos["y"] * 1000.0
    z_mm = pos["z"] * 1000.0

    # ---- quaternion ----
    qx = ori["x"]
    qy = ori["y"]
    qz = ori["z"]
    qw = ori["w"]

    # ---- optional: Euler (for display only) ----
    roll, pitch, yaw = quaternion_to_euler_deg(qx, qy, qz, qw)

    # ---- return SAME FORMAT as real function ----
    return {
        "x": x_mm,
        "y": y_mm,
        "z": z_mm,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw
    }


def cup_local_to_base(ee_pose, cup_local_mm):
    """
    ee_pose: dict with x,y,z,qx,qy,qz,qw in base frame
    cup_local_mm: np.array([x,y,z]) in camera/gripper local frame, unit mm
    """
    """
    Convert cup position from camera/gripper local frame into robot base frame.

    Input:
        ee_pose:
            current robot gripper pose in base frame
        cup_local_mm:
            cup position in local camera/gripper frame

    Output:
        cup_base:
            cup position in robot base frame (mm)

    Main idea:
        cup_base = ee_position + rotated_local_vector

    This function uses:
    1) R_robot:
       rotation from /ee_pose quaternion
    2) R_offset:
       fixed axis mapping between camera/gripper frame and base frame

    So this function is the core transformation step of the whole project.
    """


    # IMPORTANT:
    # This matrix depends on how the camera/gripper axes are defined.
    # If the axis definition changes later, this matrix must be checked again.
        
    # --- Robot rotation (from /ee_pose quaternion) ---
    R_robot = quaternion_to_rotation_matrix(
        ee_pose["qx"],
        ee_pose["qy"],
        ee_pose["qz"],
        ee_pose["qw"]
    )

    # --- FIXED rotation: gripper → base ---
    # Mapping:
    # camera X (down)   → base -Z
    # camera Y (left)   → base +Y
    # camera Z (forward)→ base +X
    R_offset = np.array([
        [0, 0, 1],    # Z_cam → X_base
        [0, 1, 0],    # Y_cam → Y_base
        [-1, 0, 0]    # X_cam → -Z_base
    ], dtype=np.float64)

    # --- Position of end-effector in base ---
    ee_pos = np.array([
        ee_pose["x"],
        ee_pose["y"],
        ee_pose["z"]
    ], dtype=np.float64)

    # --- FULL TRANSFORMATION ---
    cup_rotated = R_robot @ cup_local_mm
    cup_base = ee_pos + cup_rotated

    return cup_base

def pixel_to_reference_plane(u, v, camera_matrix, dist_coeffs, rvec, tvec):
    """
    Convert one image pixel into a 3D point on the ArUco marker plane.

    Input:
        (u, v) = image pixel of the cup point
        camera_matrix, dist_coeffs = camera calibration
        rvec, tvec = ArUco marker pose relative to camera

    Output:
        3D point on marker plane (marker frame)

    Why this is needed:
    - YOLO gives only 2D image coordinates
    - We need a real 3D reference
    - ArUco marker defines a known plane in the real world
    - This function projects the image point onto that plane
    """
    
    pixel = np.array([[[u, v]]], dtype=np.float32)

    undistorted = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
    x = float(undistorted[0, 0, 0])
    y = float(undistorted[0, 0, 1])

    ray_dir = np.array([x, y, 1.0], dtype=np.float64)
    ray_dir /= np.linalg.norm(ray_dir)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    plane_normal = R[:, 2]

    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-9:
        return None

    s = np.dot(plane_normal, t) / denom
    if s < 0:
        return None

    P_cam = s * ray_dir
    P_board = R.T @ (P_cam - t)

    return P_board


def board_to_camera_point(P_board, rvec, tvec):
    """
    Convert a 3D point from reference frame to camera frame.

    Why this is needed:
    - After finding a point on the ArUco marker plane,
      we sometimes want to express it in camera coordinates
    - The camera/gripper local position is built from camera frame
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    P_board = np.array(P_board, dtype=np.float64).reshape(3)
    P_cam = R @ P_board + t
    return P_cam


def camera_cv_to_user_frame(P_cam):
    """
    Convert OpenCV camera frame into the custom frame used in this project.

    OpenCV frame:
        X_cv = right
        Y_cv = down
        Z_cv = forward

    Custom project frame:
        X = down
        Y = left
        Z = forward

    This function changes axis order/sign so the values match
    the frame convention used in the rest of the robot project.
    """
    X_cv, Y_cv, Z_cv = P_cam

    X_user = Y_cv        # down
    Y_user = -X_cv       # left
    Z_user = Z_cv        # forward

    return np.array([X_user, Y_user, Z_user], dtype=np.float64)


def camera_point_to_ground_local_frame(P_cam, rvec, tvec):
    """
    Convert camera-frame point into the project local frame:
    X = down toward ArUco/ground plane, Y = left, Z = horizontal forward.
    """
    R, _ = cv2.Rodrigues(rvec)
    plane_normal = R[:, 2].astype(np.float64)
    plane_point = tvec.reshape(3).astype(np.float64)

    signed_height = float(np.dot(plane_normal, plane_point))
    if abs(signed_height) < 1e-9:
        return camera_cv_to_user_frame(P_cam)

    down_axis = plane_normal * np.sign(signed_height)
    down_axis /= np.linalg.norm(down_axis)

    camera_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_axis = camera_forward - np.dot(camera_forward, down_axis) * down_axis
    forward_norm = np.linalg.norm(forward_axis)
    if forward_norm < 1e-9:
        return camera_cv_to_user_frame(P_cam)
    forward_axis /= forward_norm

    left_axis = np.cross(forward_axis, down_axis)
    left_axis /= np.linalg.norm(left_axis)

    P_cam = np.array(P_cam, dtype=np.float64).reshape(3)
    return np.array([
        float(np.dot(P_cam, down_axis)),
        float(np.dot(P_cam, left_axis)),
        float(np.dot(P_cam, forward_axis))
    ], dtype=np.float64)


def scale_camera_matrix_for_image(camera_matrix, scale):
    """
    Scale camera intrinsics when the image is resized for detection.

    This lets us run ArUco detection on a larger temporary image while
    keeping the returned pose in the same real-world units.
    """
    scaled = camera_matrix.copy()
    scaled[0, 0] *= scale
    scaled[1, 1] *= scale
    scaled[0, 2] *= scale
    scaled[1, 2] *= scale
    return scaled


def unscale_corners(corners, scale):
    if corners is None or scale == 1.0:
        return corners
    if isinstance(corners, tuple):
        return tuple(corner / scale for corner in corners)
    if isinstance(corners, list):
        return [corner / scale for corner in corners]
    return corners / scale


def get_aruco_marker_object_points(marker_length):
    half = float(marker_length) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32
    )


def detect_aruco_pose(frame, camera_matrix, dist_coeffs):

    """
    Detect the selected ArUco marker in the image and estimate its pose.

    Output:
        ok_pose = whether pose estimation succeeded
        rvec, tvec = marker pose relative to camera

    Why this is important:
    - The ArUco marker is the real-world reference in this project
    - Without it, we cannot convert image positions into meaningful 3D positions
    """

    detect_scale = float(ARUCO_DETECT_SCALE)
    if detect_scale > 1.0:
        detect_frame = cv2.resize(
            frame,
            None,
            fx=detect_scale,
            fy=detect_scale,
            interpolation=cv2.INTER_CUBIC
        )
        pose_camera_matrix = scale_camera_matrix_for_image(camera_matrix, detect_scale)
    else:
        detect_scale = 1.0
        detect_frame = frame
        pose_camera_matrix = camera_matrix

    gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, _ = detect_aruco_markers(gray)

    if marker_ids is None or len(marker_ids) == 0:
        return False, None, None, None, None

    marker_ids_flat = marker_ids.reshape(-1)
    target_indexes = np.where(marker_ids_flat == ARUCO_MARKER_ID)[0]
    scaled_marker_corners = unscale_corners(marker_corners, detect_scale)
    if len(target_indexes) == 0:
        return False, None, None, scaled_marker_corners, marker_ids

    target_index = int(target_indexes[0])
    image_points = np.asarray(marker_corners[target_index], dtype=np.float32).reshape(4, 2)
    object_points = get_aruco_marker_object_points(ARUCO_MARKER_LENGTH)

    ok_pose = False
    rvec = None
    tvec = None
    for pnp_flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok_pose, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                pose_camera_matrix,
                dist_coeffs,
                flags=pnp_flag
            )
        except cv2.error:
            ok_pose = False

        if ok_pose:
            break

    return (
        ok_pose,
        rvec,
        tvec,
        scaled_marker_corners,
        marker_ids
    )


def get_best_cup_detection(results, model_names, conf_thres):

    """
    Choose the best detected cup from YOLO results.

    Rule:
    - Only keep detections with class = cup
    - Only keep detections above confidence threshold
    - If multiple cups are found, choose the one with highest confidence

    Output:
        best bounding box for the cup
    """
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


def fallback_point_from_yolo_box(x1, y1, x2, y2):
    """
    Fallback method to estimate cup point using only YOLO box.

    It uses:
    - box center in X
    - lower part of box in Y

    Why this exists:
    - contour-based lower point may fail sometimes
    - this gives a simple backup point instead of losing the cup completely
    """
    px = int((x1 + x2) / 2)
    py = int(y1 + FALLBACK_Y_RATIO * (y2 - y1))
    return px, py


def find_lower_cup_point_in_roi(frame, x1, y1, x2, y2):

    """
    Find a better lower point of the cup inside the YOLO bounding box.

    Steps:
    1) crop ROI from YOLO box
    2) convert to grayscale
    3) blur image
    4) detect edges with Canny
    5) clean edges with morphology
    6) find contours
    7) choose best contour
    8) estimate lower cup point

    Why this is important:
    - YOLO box alone is not accurate enough for a grasp/base point
    - We want a point closer to the real bottom part of the cup
    """
    h_img, w_img = frame.shape[:2]

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w_img, x2)
    y2c = min(h_img, y2)

    if x2c <= x1c or y2c <= y1c:
        return None, None, None, None

    roi = frame[y1c:y2c, x1c:x2c].copy()
    if roi.size == 0:
        return None, None, None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    search_start_y = int(gray.shape[0] * SEARCH_START_RATIO)
    gray_search = gray[search_start_y:, :]

    if gray_search.size == 0:
        return None, None, None, None

    blur = cv2.GaussianBlur(gray_search, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, edges, None

    roi_h, roi_w = gray_search.shape[:2]
    roi_cx = roi_w / 2.0

    best_contour = None
    best_score = -1e9

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if h < MIN_CONTOUR_HEIGHT:
            continue
        if w < MIN_CONTOUR_WIDTH:
            continue
        if w > roi_w * 0.95:
            continue

        cnt_cx = x + w / 2.0
        cnt_cy = y + h / 2.0

        center_penalty = abs(cnt_cx - roi_cx)
        bottom_bonus = cnt_cy

        score = area + 2.0 * bottom_bonus - 2.5 * center_penalty

        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        return None, None, edges, None

    debug_contour = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_contour, [best_contour], -1, (0, 255, 0), 2)

    pts = best_contour.reshape(-1, 2)
    ys = pts[:, 1]

    y_min = np.min(ys)
    y_max = np.max(ys)
    contour_h = y_max - y_min

    if contour_h < 5:
        return None, None, edges, debug_contour

    lower_threshold = y_min + LOWER_REGION_RATIO * contour_h
    lower_pts = pts[ys >= lower_threshold]

    if len(lower_pts) == 0:
        return None, None, edges, debug_contour

    lower_x = int(np.mean(lower_pts[:, 0]))
    lower_y = int(np.max(lower_pts[:, 1]))

    px = x1c + lower_x
    py = y1c + search_start_y + lower_y

    for p in lower_pts:
        cv2.circle(debug_contour, tuple(p), 1, (0, 0, 255), -1)
    cv2.circle(debug_contour, (lower_x, lower_y), 5, (255, 0, 255), -1)

    return px, py, edges, debug_contour


def create_cup_3d_box_points(Xc, Yc, cup_diameter, cup_height):

    """
    Create 8 corner points of a simple 3D box around the cup.

    Why this exists:
    - only for visualization/debug
    - helps show estimated 3D location of the cup in the image
    - does not control the robot directly
    """
    d = cup_diameter / 2.0
    h = CUP_HEIGHT_AXIS_SIGN * cup_height

    points_3d = np.array([
        [Xc - d, Yc - d, 0.0],
        [Xc + d, Yc - d, 0.0],
        [Xc + d, Yc + d, 0.0],
        [Xc - d, Yc + d, 0.0],
        [Xc - d, Yc - d, h],
        [Xc + d, Yc - d, h],
        [Xc + d, Yc + d, h],
        [Xc - d, Yc + d, h],
    ], dtype=np.float32)

    return points_3d


def draw_3d_box(display, points_3d, rvec, tvec, camera_matrix, dist_coeffs):


    """
    Draw the 3D cup box onto the image.

    Steps:
    - project 3D box points into 2D image points
    - draw box edges on the display image

    Why this is useful:
    - visual debug
    - helps check whether pose estimation and cup position look reasonable
    """
    if points_3d is None or len(points_3d) != 8:
        return False

    if not np.all(np.isfinite(points_3d)):
        return False

    try:
        imgpts, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    except cv2.error:
        return False

    imgpts = imgpts.reshape(-1, 2)

    if not np.all(np.isfinite(imgpts)):
        return False

    imgpts = np.round(imgpts).astype(np.int32)

    def pt(i):
        return (int(imgpts[i][0]), int(imgpts[i][1]))

    cv2.line(display, pt(0), pt(1), (0, 255, 0), 2)
    cv2.line(display, pt(1), pt(2), (0, 255, 0), 2)
    cv2.line(display, pt(2), pt(3), (0, 255, 0), 2)
    cv2.line(display, pt(3), pt(0), (0, 255, 0), 2)

    cv2.line(display, pt(4), pt(5), (0, 255, 0), 2)
    cv2.line(display, pt(5), pt(6), (0, 255, 0), 2)
    cv2.line(display, pt(6), pt(7), (0, 255, 0), 2)
    cv2.line(display, pt(7), pt(4), (0, 255, 0), 2)

    for i in range(4):
        cv2.line(display, pt(i), pt(i + 4), (0, 255, 0), 2)

    return True


def update_smoothed_fps(prev_time, prev_fps):
    now = time.time()
    if prev_time is None:
        return now, 0.0

    dt = now - prev_time
    instant_fps = 0.0 if dt <= 1e-9 else 1.0 / dt
    if prev_fps <= 0.0:
        smoothed_fps = instant_fps
    else:
        smoothed_fps = (FPS_SMOOTHING * prev_fps) + ((1.0 - FPS_SMOOTHING) * instant_fps)
    return now, smoothed_fps


def process_snapshot_candidate(frame, depth_frame, latest_ee_pose):
    display = frame.copy()
    edge_view = np.zeros((240, 320), dtype=np.uint8)
    contour_view = np.zeros((240, 320, 3), dtype=np.uint8)

    result = {
        "score": 0.0,
        "display": display,
        "edge_view": edge_view,
        "contour_view": contour_view,
        "publish_data": None,
        "cup_local_mm": None,
        "summary": "No cup detected in snapshot.",
    }

    ok_pose, rvec, tvec, marker_corners, marker_ids = detect_aruco_pose(
        frame,
        camera_matrix,
        dist_coeffs
    )

    if SHOW_TEXT_ON_MAIN_WINDOW:
        if ok_pose:
            cv2.putText(display, "ArUco pose: OK", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "ArUco pose: NOT FOUND", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    results = model(
        frame,
        verbose=False,
        device=YOLO_DEVICE,
        imgsz=YOLO_IMGSZ
    )
    best_cup = get_best_cup_detection(results, model.names, YOLO_CONF_THRES)
    if best_cup is None:
        if SHOW_TEXT_ON_MAIN_WINDOW:
            cv2.putText(display, "No cup detected", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result

    x1, y1, x2, y2, conf = best_cup
    result["score"] = 1.0 + conf
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

    contour_px = contour_py = None
    debug_edges = debug_contour = None
    if ok_pose:
        contour_px, contour_py, debug_edges, debug_contour = find_lower_cup_point_in_roi(frame, x1, y1, x2, y2)

    if debug_edges is not None:
        edge_view = debug_edges
        result["edge_view"] = edge_view
    if debug_contour is not None:
        contour_view = debug_contour
        result["contour_view"] = contour_view

    if contour_px is not None and contour_py is not None:
        cup_px, cup_py = contour_px, contour_py
        point_mode = "contour"
    else:
        cup_px, cup_py = fallback_point_from_yolo_box(x1, y1, x2, y2)
        point_mode = "fallback"

    cv2.circle(display, (cup_px, cup_py), 6, (0, 0, 255), -1)

    if not ok_pose:
        result["summary"] = "Cup detected, but ArUco pose was not found."
        return result

    result["score"] = 2.0 + conf
    cv2.drawFrameAxes(
        display,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        0.05
    )

    measurement_mode = point_mode
    depth_mm = None
    use_depth_grasp = False
    grasp_cam_from_depth = None
    P_board_for_box = pixel_to_reference_plane(
        cup_px,
        cup_py,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec
    )

    if USE_DEPTH_ASSIST:
        grasp_cam_from_depth, depth_mm = estimate_grasp_point_from_depth(
            depth_frame,
            cup_px,
            cup_py,
            camera_matrix,
            cup_box=(x1, y1, x2, y2)
        )

        if grasp_cam_from_depth is not None:
            use_depth_grasp = True

    box_anchor_ok = P_board_for_box is not None and np.all(np.isfinite(P_board_for_box))
    if box_anchor_ok:
        Xb, Yb, Zb = P_board_for_box
        box_anchor_ok = abs(Xb) <= 2.0 and abs(Yb) <= 2.0 and abs(Zb) <= 0.2

    grasp_cam = None
    ok_box = False
    if box_anchor_ok:
        cup_box_points_3d = create_cup_3d_box_points(
            Xb,
            Yb,
            CUP_DIAMETER,
            CUP_HEIGHT
        )
        ok_box = draw_3d_box(
            display,
            cup_box_points_3d,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

    if box_anchor_ok:
        grasp_board = np.array(
            [Xb, Yb, CUP_HEIGHT_AXIS_SIGN * GRASP_HEIGHT_RATIO * CUP_HEIGHT],
            dtype=np.float64
        )
        grasp_cam = board_to_camera_point(grasp_board, rvec, tvec)
        measurement_mode = f"{point_mode} + aruco"
    elif use_depth_grasp:
        grasp_cam = grasp_cam_from_depth
        measurement_mode = f"{point_mode} + depth fallback"
    else:
        measurement_mode = f"{point_mode} + no 3D"

    if grasp_cam is None or not np.all(np.isfinite(grasp_cam)):
        result["summary"] = "Cup and ArUco found, but 3D point could not be computed."
        return result

    result["score"] = 3.0 + conf
    cup_user = camera_point_to_ground_local_frame(grasp_cam, rvec, tvec)
    cup_local_mm = cup_user * 1000.0
    cup_local_mm[0] += X_OFFSET_CAMERA_MM
    cup_local_mm[1] += Y_OFFSET_CAMERA_MM
    cup_local_mm[2] += Z_OFFSET_CAMERA_MM
    result["cup_local_mm"] = cup_local_mm

    X_mm = cup_local_mm[0]
    Y_mm = cup_local_mm[1]
    Z_mm = cup_local_mm[2]

    if latest_ee_pose is not None:
        cup_base_mm = cup_local_to_base(latest_ee_pose, cup_local_mm)
        base_x = cup_base_mm[0]
        base_y = cup_base_mm[1]
        base_z = cup_base_mm[2]
        result["publish_data"] = (base_x, base_y, base_z)
        result["summary"] = (
            f"LOCAL(mm) X={X_mm:.1f}, Y={Y_mm:.1f}, Z={Z_mm:.1f} | "
            f"BASE(mm) X={base_x:.1f}, Y={base_y:.1f}, Z={base_z:.1f}"
        )
        result["score"] = 4.0 + conf
    else:
        result["summary"] = (
            f"LOCAL(mm) X={X_mm:.1f}, Y={Y_mm:.1f}, Z={Z_mm:.1f} | No /ee_pose"
        )

    if SHOW_TEXT_ON_MAIN_WINDOW:
        cv2.putText(
            display,
            f"Local X={X_mm:.1f}mm Y={Y_mm:.1f}mm Z={Z_mm:.1f}mm",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )
        if latest_ee_pose is not None and result["publish_data"] is not None:
            base_x, base_y, base_z = result["publish_data"]
            cv2.putText(
                display,
                f"Base X={base_x:.1f}mm Y={base_y:.1f}mm Z={base_z:.1f}mm",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )
        cv2.putText(
            display,
            f"mode={measurement_mode}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        if depth_mm is not None:
            cv2.putText(
                display,
                f"Depth={depth_mm:.0f} mm",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                2
            )
        elif not ok_box:
            cv2.putText(
                display,
                "3D box skipped",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

    return result


# =========================
# MAIN
# =========================

# Snapshot flow:
# 1) Warm up camera and keep the latest valid frame
# 2) Detect ArUco marker once
# 3) Detect cup with YOLO once
# 4) Find lower cup point
# 5) Convert image point -> board -> camera -> local frame
# 6) Convert local cup position -> base frame
# 7) Publish one result
# 8) Show one result image

if not HEADLESS:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

if SHOW_DEBUG_WINDOWS and not HEADLESS:
    cv2.namedWindow("Cup ROI Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cup ROI Contour", cv2.WINDOW_NORMAL)

print("Capturing snapshot...")

timing_marks = [("start", time.time())]

ok, snapshot_candidates = capture_snapshot_frames()
if not ok:
    print("Failed to capture snapshot frame.")
    if webcam_cap is not None:
        webcam_cap.release()
    if hp60_camera is not None:
        hp60_camera.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
    raise SystemExit(1)

timing_marks.append(("capture done", time.time()))

best_result = None
best_index = None
print(f"Processing {len(snapshot_candidates)} snapshot candidates...")
for idx, (frame, depth_frame) in enumerate(snapshot_candidates, start=1):
    candidate_result = process_snapshot_candidate(frame, depth_frame, None)
    print(
        f"Snapshot {idx}/{len(snapshot_candidates)} | "
        f"score={candidate_result['score']:.2f} | "
        f"{candidate_result['summary']}"
    )
    if best_result is None or candidate_result["score"] >= best_result["score"]:
        best_result = candidate_result
        best_index = idx

timing_marks.append(("process/select done", time.time()))

display = best_result["display"]
edge_view = best_result["edge_view"]
contour_view = best_result["contour_view"]

print(f"Selected snapshot {best_index}/{len(snapshot_candidates)}")
print(best_result["summary"])

if best_result["cup_local_mm"] is not None:
    latest_ee_pose, pose_source = wait_for_one_ee_pose(WS_URL)
    timing_marks.append(("ee_pose done", time.time()))
    print(f"EE pose source: {pose_source}")

    if latest_ee_pose is not None:
        cup_local_mm = best_result["cup_local_mm"]
        cup_base_mm = cup_local_to_base(latest_ee_pose, cup_local_mm)
        base_x = float(cup_base_mm[0])
        base_y = float(cup_base_mm[1])
        base_z = float(cup_base_mm[2])
        best_result["publish_data"] = (base_x, base_y, base_z)

        print(
            f"BASE(mm) X={base_x:.1f}, Y={base_y:.1f}, Z={base_z:.1f}"
        )

        if SHOW_TEXT_ON_MAIN_WINDOW:
            cv2.putText(
                display,
                f"Base X={base_x:.1f}mm Y={base_y:.1f}mm Z={base_z:.1f}mm",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

        published = publish_result_once(base_x, base_y, base_z)
        timing_marks.append(("publish done", time.time()))
        print("Publish result:", "sent" if published else "not sent")

        if TASK_PLANNING_PUBLISH:
            initial_published = publish_task_predicates_once(TASK_INITIAL_TOPIC, TASK_INITIAL_STATE)
            timing_marks.append(("task initial publish done", time.time()))
            print(
                f"Published {TASK_INITIAL_TOPIC}:",
                TASK_INITIAL_STATE if initial_published else "not sent"
            )

            goal_published = publish_task_predicates_once(TASK_GOAL_TOPIC, TASK_GOAL_STATE)
            timing_marks.append(("task goal publish done", time.time()))
            print(
                f"Published {TASK_GOAL_TOPIC}:",
                TASK_GOAL_STATE if goal_published else "not sent"
            )

            state_published = publish_task_predicates_once(TASK_STATE_TOPIC, SEARCH_CUP_SUCCESS_STATE)
            timing_marks.append(("task state publish done", time.time()))
            print(
                f"Published {TASK_STATE_TOPIC}:",
                SEARCH_CUP_SUCCESS_STATE if state_published else "not sent"
            )
        else:
            print("Task planning publish disabled.")
    else:
        timing_marks.append(("ee_pose timeout", time.time()))
        print("No /ee_pose received before timeout. Result was not published.")
else:
    timing_marks.append(("no local result", time.time()))
    print("No valid LOCAL result. Result was not published.")

if SAVE_DEBUG_IMAGE or HEADLESS:
    cv2.imwrite(DEBUG_IMAGE_PATH, display)
    print("Saved debug image:", DEBUG_IMAGE_PATH)

if not HEADLESS:
    cv2.imshow(WINDOW_NAME, display)

if SHOW_DEBUG_WINDOWS and not HEADLESS:
    cv2.imshow("Cup ROI Edges", edge_view)
    cv2.imshow("Cup ROI Contour", contour_view)

timing_marks.append(("screen ready", time.time()))
print_duration_report(timing_marks)

if not HEADLESS:
    print("Snapshot ready. Press q or Esc to close.")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break
else:
    print("Headless snapshot complete.")

if webcam_cap is not None:
    webcam_cap.release()
if hp60_camera is not None:
    hp60_camera.stop()
if not HEADLESS:
    cv2.destroyAllWindows()
