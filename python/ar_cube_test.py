"""
Hai tay dieu khien cube:
- Tay GRAB_HAND_LABEL: pinch (ngon cai + ngon tro) gan cube -> cam va di
  chuyen ca 3 truc X/Y (ngang/doc man hinh) va Z (day/keo lai gan-xa camera).
- Tay ROTATE_HAND_LABEL: nam tay (fist) -> xoay cube quanh truc X va Y theo
  chuyen dong cua tay.
- Cube dung im khi khong tay nao tuong tac (khong con tu xoay).
- Vi tri tay duoc loc qua One Euro Filter (loc rung khi dung yen, bam sat
  ngay khi di chuyen nhanh) de chuyen dong muot ma khong bi tre.
- Co debounce + hysteresis rieng cho tung cu chi de chong tuot/nhay trang thai.

pip install moderngl pyrr numpy opencv-python mediapipe
"""
import argparse
from pathlib import Path
from stl import mesh

import math

import time
from collections import deque

import cv2
import mediapipe as mp
import moderngl
import numpy as np
import pyrr

WIDTH, HEIGHT = 640, 480

# ==== Camera/projection ====
FOV_DEG = 45.0
EYE_Z = 6.0
NEAR, FAR = 0.1, 100.0

# ==== Nguong pinch: ty le khoang cach ngon cai-tro / kich thuoc long ban tay ====
# Chinh 2 so nay neu thay kho/de pinch qua so voi tay/webcam cua ban.
PINCH_ENTER = 0.35   # duoi nguong nay -> coi la dang pinch
PINCH_EXIT = 0.55    # tren nguong nay -> coi la tha pinch

RATIO_SMOOTH_WINDOW = 5
DEBOUNCE_FRAMES = 3  # can bay nhieu frame lien tiep dat dieu kien moi doi trang thai

GRAB_RADIUS_PX = 320
ROTATE_SENSITIVITY = 0.012  # radian xoay them moi pixel tay xoay di chuyen ngang

Z_SENSITIVITY = 3.5   # he so quy doi ty le thay doi kich thuoc ban tay sang world units
Z_MIN, Z_MAX = -3.0, 4.0  # gioi han de vat the khong lot qua camera (EYE_Z=6)

MOVE_DEAD_ZONE_PX = 2.0
ROTATE_DEAD_ZONE_PX = 2.0
DEPTH_DEAD_ZONE = 0.005
# Rotate chi kich hoat khi ca ngon tro VA ngon giua deu cham gan ngon cai.
# Nguong cu rong qua, pinch 2 ngon thuong bi nham thanh rotate.
THREE_PINCH_ENTER = 0.24
THREE_PINCH_EXIT = 0.38
# Nam tay can nhay hon mot chut; hysteresis lon giup khong chop tat.
# Cao hon nguong fist-exit de pinch/three-finger pinch khong bi xem la xoe tay.
OPEN_HAND_DEBOUNCE_FRAMES = 4
MODEL_SELECTOR_STEP_RAD = math.radians(35)
MODEL_SELECTOR_HOLD_SECONDS = 3.0

# Doi 2 gia tri nay neu muon dao vai tro tay trai/phai
GRAB_HAND_LABEL = "Right"
ROTATE_HAND_LABEL = "Left"

# Tat ca cac manh Saturn V phai dung CUNG MOT scale.  Khong auto-scale tung
# file STL, vi nhu vay moi tang deu co cung kich thuoc va bi tach roi nhau.
# Gan day chieu cao khung nhin (xap xi 430 px trong framebuffer 480 px).
ROCKET_WORLD_HEIGHT = 4.5
# Truc ao cua Saturn V: moi tam than/cut-plane deu nam tren duong nay.
ROCKET_AXIS_X = 0.0
ROCKET_AXIS_Z = 0.0
# Thu tu STL la S-IC (day) -> escape tower (dinh), cung chieu truc doc camera.
# Gia tri +1 dat phan day rong o duoi va tower manh o tren nhu Saturn V that.
ROCKET_VERTICAL_FLIP = 1.0
# Hai mesh phan dau co bounding box lech mat cat thuc. Day la trim tren truc
# ao de command module va escape tower dung dung tren service module.
ROCKET_AXIS_Y_TRIM = {
    # Chi trim rat nhe; trim lon se tao khe ho giua service module, command
    # module va escape tower khi nhin chinh dien.
    "command moduel.stl": 0.10,
    "escape tower.stl": 0.15,
}
# Bounding box cua vai STL co khoang trong o dau tang. Cho cac tang chen nhe
# vao nhau de khi ghep khong con khe ho.
ROCKET_JOIN_OVERLAP = 0.12
# Cac vi tri tách "ngau nhien co kiem soat": moi slot cach nhau du xa de
# cac STL lon khong cham nhau, nhung khong thanh hang/luoi qua deu.
EXPLODED_SLOTS = (
    (-1.42, 1.02, -0.18), (-0.10, 1.42, 0.16), (1.14, 0.83, -0.10),
    (-1.50, 0.08, 0.18), (-0.29, 0.33, -0.16), (1.28, -0.22, 0.12),
    (-0.81, -1.23, 0.08), (0.45, -1.42, -0.16), (1.47, -1.00, 0.18),
)
ROCKET_PART_ORDER = (
    "s-ic bottom.stl",
    "s-ic top.stl",
    "stage 1-2 coupler v2.stl",
    "s-ii.stl",
    "s-iv b.stl",
    "lem shroud.stl",
    "service module.stl",
    "command moduel.stl",
    "escape tower.stl",
)

# Model rover hien thi rieng, khong phai la mot manh cua Saturn V.
ROVER_MODEL_FILENAME = "Curiosity Rover (MSL) (Clean).stl"
ROVER_WORLD_SIZE = 3.0
ROMAN_MODEL_FILENAME = "Nancy Grace Roman Space Telescope (1).stl"
ROMAN_WORLD_SIZE = 3.2
# Roman co nhieu tam giac nho lien ket; lay mau thua lam be vo be mat.
# Giu day du mesh de cac manh cua kinh thien van lien tuc.
ROMAN_MAX_TRIANGLES = None
MODEL_CHOICES = ("rover", "saturn", "roman")
# STL rover co kem 12 triangle tao thanh mot khoi lap phuong 1x1 khong thuoc
# rover. Cac mat rover lon nhat nho hon 0.31, nen nguong nay chi loai khoi do.
ROVER_PLACEHOLDER_CUBE_AREA = 0.49


def selected_model_mode():
    parser = argparse.ArgumentParser(description="Gesture-controlled 3D model viewer")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="rover",
        help="Model to show: rover (default), saturn, or roman.",
    )
    return parser.parse_known_args()[0].model


ACTIVE_MODEL_MODE = selected_model_mode()


class Model:
    """Small STL-backed ModernGL model wrapper with transform state."""

    def __init__(
        self,
        ctx,
        program,
        mesh_path,
        position=None,
        rotation=None,
        scale=None,
        local_y_flip=True,
        local_y_rotation=0.0,
        local_x_rotation=0.0,
        max_triangles=None,
    ):
        self.ctx = ctx
        self.program = program
        self.mesh_path = mesh_path
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array(rotation if rotation is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        # File STL Saturn V dung truc Y nguoc voi truc Y cua scene.  Luu
        # transform nay tren tung mesh (thay vi dao ca cum part) de dau
        # rocket luon huong len va moi noi van dung dung vi tri.
        self.local_y_flip = local_y_flip
        self.local_y_rotation = local_y_rotation
        self.local_x_rotation = local_x_rotation
        self.max_triangles = max_triangles
        self.mesh = self._load_mesh(mesh_path)
        self.center = np.zeros(3, dtype=np.float32)
        if scale is None:
            self._auto_scale()
        else:
            # Van can giua mesh ngay ca khi dung scale chung cho Saturn V.
            positions = self.mesh.vectors.astype(np.float32).reshape(-1, 3)
            if positions.size:
                self.center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
            self.scale = np.array(scale, dtype=np.float32)
        self.vbo, self.ibo, self.vao = self._build_gl_buffers()

    def _load_mesh(self, mesh_path):
        candidates = []
        if mesh_path:
            raw_path = str(mesh_path)
            candidates.append(Path(raw_path))
            candidates.append(Path(__file__).resolve().parent / raw_path)
        candidates.extend([
            Path(__file__).resolve().parent / "models/command moduel.stl",
            Path("models/command moduel.stl"),
            Path(__file__).resolve().parent / "command moduel.stl",
            Path("command moduel.stl"),
        ])

        for candidate in candidates:
            if candidate.exists():
                try:
                    return mesh.Mesh.from_file(str(candidate))
                except Exception:
                    continue

        return self._create_fallback_mesh()

    @staticmethod
    def _create_fallback_mesh():
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ], dtype=np.int32)
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        stl_mesh.vectors = vertices[faces]
        return stl_mesh

    def _auto_scale(self):
        positions = self.mesh.vectors.astype(np.float32).reshape(-1, 3)
        if positions.size == 0:
            self.scale = np.ones(3, dtype=np.float32)
            self.center = np.zeros(3, dtype=np.float32)
            return

        min_xyz = positions.min(axis=0)
        max_xyz = positions.max(axis=0)
        self.center = (min_xyz + max_xyz) / 2.0
        extent = np.max(max_xyz - min_xyz)
        if extent <= 1e-6:
            self.scale = np.ones(3, dtype=np.float32)
        else:
            auto_scale = 1.2 / extent
            self.scale = np.full(3, auto_scale, dtype=np.float32)

    def _build_gl_buffers(self):
        triangles = self.mesh.vectors.astype(np.float32)
        if Path(self.mesh_path).name.lower() == ROVER_MODEL_FILENAME.lower():
            double_areas = np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            triangles = triangles[double_areas / 2 < ROVER_PLACEHOLDER_CUBE_AREA]
        if self.max_triangles is not None and len(triangles) > self.max_triangles:
            stride = math.ceil(len(triangles) / self.max_triangles)
            triangles = triangles[::stride]
        positions = (triangles.reshape(-1, 3) - self.center).astype(np.float32)
        if self.local_y_flip:
            positions[:, 1] *= -1.0
        if self.local_y_rotation:
            cosine = math.cos(self.local_y_rotation)
            sine = math.sin(self.local_y_rotation)
            x = positions[:, 0].copy()
            z = positions[:, 2].copy()
            positions[:, 0] = x * cosine + z * sine
            positions[:, 2] = -x * sine + z * cosine
        if self.local_x_rotation:
            cosine = math.cos(self.local_x_rotation)
            sine = math.sin(self.local_x_rotation)
            y = positions[:, 1].copy()
            z = positions[:, 2].copy()
            positions[:, 1] = y * cosine - z * sine
            positions[:, 2] = y * sine + z * cosine
        positions = positions * self.scale
        colors = self._saturn_v_colors(positions)
        y_min = positions[:, 1].min()
        y_span = max(float(positions[:, 1].max() - y_min), 1e-6)
        local_height = ((positions[:, 1] - y_min) / y_span)[:, None]
        style = np.full((len(positions), 1), self._saturn_v_style(), dtype=np.float32)
        face_positions = positions.reshape(-1, 3, 3)
        normals = np.cross(
            face_positions[:, 1] - face_positions[:, 0],
            face_positions[:, 2] - face_positions[:, 0],
        )
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(normal_lengths, 1e-6)
        vertex_normals = np.repeat(normals, 3, axis=0)
        vertices = np.hstack([positions, colors, local_height, style, vertex_normals]).astype("f4")
        indices = np.arange(positions.shape[0], dtype="i4")

        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 3f 1f 1f 3f", "in_position", "in_color", "in_height", "in_style", "in_normal")],
            ibo,
        )
        return vbo, ibo, vao

    def _saturn_v_style(self):
        styles = {
            "fin (print 4).stl": 1,
            "service module.stl": 2,
            "escape tower.stl": 3,
            "command moduel.stl": 4,
            "lem shroud.stl": 5,
            "stage 1-2 coupler v2.stl": 6,
            "s-ic bottom.stl": 7,
            "s-ic top.stl": 8,
            "s-ii.stl": 9,
            "s-iv b.stl": 10,
        }
        return styles.get(Path(self.mesh_path).name.lower(), 0)

    def _saturn_v_colors(self, positions):
        # Shader dung in_height + in_style de to dai mau dung theo tung pixel.
        # Attribute mau nay giu de vertex format don gian va dung chung fallback.
        return np.full((len(positions), 3), 0.88, dtype=np.float32)

    def draw(self, mvp=None):
        if self.vao is None:
            return

        if mvp is None:
            rot_x = pyrr.matrix44.create_from_x_rotation(float(self.rotation[0]))
            rot_y = pyrr.matrix44.create_from_y_rotation(float(self.rotation[1]))
            rot_z = pyrr.matrix44.create_from_z_rotation(float(self.rotation[2]))
            rotation = pyrr.matrix44.multiply(pyrr.matrix44.multiply(rot_x, rot_y), rot_z)
            scale_mat = pyrr.matrix44.create_from_scale(np.ones(3, dtype=np.float32))
            translation = pyrr.matrix44.create_from_translation(self.position)
            model = pyrr.matrix44.multiply(pyrr.matrix44.multiply(rotation, scale_mat), translation)
            mv = pyrr.matrix44.multiply(model, view)
            mvp = pyrr.matrix44.multiply(mv, proj)

        self.program["mvp"].write(np.asarray(mvp, dtype="f4").tobytes())
        self.vao.render(moderngl.TRIANGLES)

    def render(self, mvp=None):
        self.draw(mvp)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils


def open_camera(start_index=0, max_index=4):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for device in range(start_index, start_index + max_index + 1):
        for backend in backends:
            cap = cv2.VideoCapture(device, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Opened webcam device {device} with backend {backend}.")
                return cap
            cap.release()
    return None


def distance(a, b, w, h):
    ax, ay = a.x * w, a.y * h
    bx, by = b.x * w, b.y * h
    return math.hypot(ax - bx, ay - by)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def apply_dead_zone(value, threshold):
    return 0.0 if abs(value) < threshold else value


def smooth_value(current, target, alpha):
    return current + (target - current) * alpha


def three_finger_pinch_ratio(landmarks, w, h):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = distance(wrist, middle_mcp, w, h)
    thumb_index = distance(landmarks[4], landmarks[8], w, h)
    thumb_middle = distance(landmarks[4], landmarks[12], w, h)
    return max(thumb_index, thumb_middle) / palm_size


def palm_size_px(landmarks, w, h):
    return distance(landmarks[0], landmarks[9], w, h)


def pinch_ratio(landmarks, w, h):
    """Ty le khoang cach ngon cai-tro, chuan hoa theo kich thuoc long ban tay
    (co tay-goc ngon giua) -> khong doi khi tay gan/xa camera."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = distance(wrist, middle_mcp, w, h)
    return distance(landmarks[4], landmarks[8], w, h) / palm_size

def pinch_point_px(landmarks, w, h):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    x = (thumb_tip.x + index_tip.x) / 2 * w
    y = (thumb_tip.y + index_tip.y) / 2 * h

    palm_size = palm_size_px(landmarks, w, h)
    return x, y, palm_size


def fist_ratio(landmarks, w, h):
    """Ty le trung binh khoang cach 4 dau ngon (tro, giua, ap ut, ut) toi co
    tay, chuan hoa theo long ban tay. Nho = nam chat, lon = xoe tay."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = distance(wrist, middle_mcp, w, h)
    fingertip_ids = [8, 12, 16, 20]
    total = sum(distance(wrist, landmarks[tid], w, h) for tid in fingertip_ids)
    return (total / len(fingertip_ids)) / palm_size


def is_open_hand(landmarks):
    """Nhan biet ban tay xoe bang cac ngon dang duoi, on dinh hon ti le
    khoang cach khi webcam gan/xa. Can it nhat 3 trong 4 ngon duoi."""
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = sum(landmarks[tip].y < landmarks[pip].y for tip, pip in finger_pairs)
    return extended >= 3


def palm_rotation_angle(landmarks, w, h):
    """Goc cua truc co tay -> goc ngon giua, dung de xoay scope chon model."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    return math.atan2((middle_mcp.y - wrist.y) * h, (middle_mcp.x - wrist.x) * w)


def wrapped_angle_delta(current, previous):
    return (current - previous + math.pi) % (2 * math.pi) - math.pi


def is_fist_hand(landmarks, w, h):
    """Nhan fist theo tung ngon cuon ve gan co tay.

    Cach nay khong bi tre nhu ti le trung binh va cung khong nham pinch
    cai-tro (cac ngon con lai van dang duoi) thanh fist.
    """
    palm = max(palm_size_px(landmarks, w, h), 1e-6)
    curled_tips = (8, 12, 16, 20)
    curled = sum(distance(landmarks[0], landmarks[tip], w, h) < palm * 1.65 for tip in curled_tips)
    return curled >= 3


def gesture_hand(landmark_lists, ratio_func, w, h):
    """Tra ve tay thuc hien cu chi ro nhat (ratio nho nhat)."""
    if not landmark_lists:
        return None
    return min(landmark_lists, key=lambda points: ratio_func(points, w, h))


def palm_center_px(landmarks, w, h):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    return (wrist.x + middle_mcp.x) / 2 * w, (wrist.y + middle_mcp.y) / 2 * h


class GestureState:
    """Trang thai cu chi cho 1 vai tro tay (co the la pinch hoac fist tuy
    ratio_func truyen vao): lam muot ty le qua nhieu frame, hysteresis (2
    nguong khac nhau) + debounce (can du so frame lien tiep) truoc khi thuc
    su doi trang thai -> chong tuot/nhay."""

    def __init__(self, ratio_func, point_func, enter_th, exit_th, grace_period=0.2):
        self.ratio_func = ratio_func
        self.point_func = point_func
        self.enter_th = enter_th
        self.exit_th = exit_th
        self.grace_period = grace_period
        self.ratio_history = deque(maxlen=RATIO_SMOOTH_WINDOW)
        self.active = False
        self._pending_target = None
        self._pending_count = 0
        self.missing_since = None

    def _reset(self):
        self.ratio_history.clear()
        self._pending_target = None
        self._pending_count = 0
        self.active = False
        self.missing_since = None

    def update(self, landmarks, w, h, now=None):
        now = now if now is not None else time.time()

        if landmarks is None:
            if self.active:
                if self.missing_since is None:
                    self.missing_since = now
                elif now - self.missing_since >= self.grace_period:
                    self._reset()
            else:
                self._reset()
            return None

        self.missing_since = None

        ratio = self.ratio_func(landmarks, w, h)
        self.ratio_history.append(ratio)
        smooth = sum(self.ratio_history) / len(self.ratio_history)

        target = self.active
        if self.active and smooth > self.exit_th:
            target = False
        elif not self.active and smooth < self.enter_th:
            target = True

        if target != self.active:
            if target == self._pending_target:
                self._pending_count += 1
            else:
                self._pending_target = target
                self._pending_count = 1
            if self._pending_count >= DEBOUNCE_FRAMES:
                self.active = target
                self._pending_target = None
                self._pending_count = 0
        else:
            self._pending_target = None
            self._pending_count = 0

        return self.point_func(landmarks, w, h)


# ---------------------------------------------------------------------------
# One Euro Filter: loc rung khi tay dung yen/di cham, tu no long khi tay di
# nhanh de khong bi tre. Phu hop hon moving-average don thuan cho tracking tay.
# ---------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self):
        self.t_prev = None


class VecEuroFilter:
    """Boc nhieu OneEuroFilter lai, moi truc (x, y, z, ...) 1 filter rieng."""

    def __init__(self, dims, **kwargs):
        self.filters = [OneEuroFilter(**kwargs) for _ in range(dims)]

    def filter(self, vec, t):
        return tuple(f.filter(v, t) for f, v in zip(self.filters, vec))

    def reset(self):
        for f in self.filters:
            f.reset()


# ---------------------------------------------------------------------------
# Offscreen 3D renderer (moderngl) — giu nguyen tu ban truoc
# ---------------------------------------------------------------------------
ctx = moderngl.create_standalone_context()
ctx.enable(moderngl.DEPTH_TEST)

vertex_shader = """
#version 330
in vec3 in_position;
in vec3 in_color;
in float in_height;
in float in_style;
in vec3 in_normal;
uniform mat4 mvp;
out vec3 v_color;
out float v_height;
flat out float v_style;
flat out vec3 v_normal;
out vec3 v_local_position;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
    v_height = in_height;
    v_style = in_style;
    v_normal = in_normal;
    v_local_position = in_position;
}
"""

fragment_shader = """
#version 330
in vec3 v_color;
in float v_height;
flat in float v_style;
flat in vec3 v_normal;
in vec3 v_local_position;
out vec4 f_color;

bool band(float start, float end) {
    return v_height >= start && v_height <= end;
}

void main() {
    vec3 white = vec3(0.88, 0.88, 0.85);
    vec3 black = vec3(0.10, 0.11, 0.12);
    vec3 metal = vec3(0.38, 0.40, 0.42);
    // Giữ in_color la attribute active de vertex format cua mesh va fallback
    // cube dung chung mot layout; he so nho nay khong anh huong mau hien thi.
    vec3 color = mix(white, v_color, 0.0001);

    // To theo pixel tren truc doc cua moi STL: dai mau luon ngang va sac,
    // khong bi rang cua theo cac triangle dai cua mesh.
    if (v_style == 1.0) {
        // Bon canh on dinh o day S-IC la mau trang.
        color = white;
    } else if (v_style == 2.0 || v_style == 3.0 || v_style == 4.0) {
        // Service module, command module va escape tower de trang sach.
        color = white;
    } else if (v_style == 5.0) {
        color = band(0.00, 0.22) ? black : white;
    } else if (v_style == 6.0) {
        color = band(0.36, 0.64) ? white : black;
    } else if (v_style == 7.0) {
        // 1/4 phan duoi S-IC: bon o den cung kich thuoc quanh than.  Vi
        // tinh theo goc cua pixel, cac o van dung vi tri khi rocket xoay.
        float tail_angle = atan(v_local_position.z, v_local_position.x);
        bool one_of_four_tail_panels = abs(sin(2.0 * tail_angle)) > 0.72;
        // Cum dong co/nozzle nam o sat day (mau xam kim loai). Bon mang den
        // bat dau phia tren no, nen khong cat ngang hay lam mat nozzle.
        if (band(0.00, 0.10)) {
            color = metal;
        } else if (band(0.10, 0.25) && one_of_four_tail_panels) {
            color = black;
        } else {
            color = white;
        }
    } else if (v_style == 8.0) {
        color = (band(0.12, 0.18) || band(0.79, 0.84)) ? black : white;
    } else if (v_style == 9.0) {
        color = (band(0.08, 0.15) || band(0.73, 0.80)) ? black : white;
    } else if (v_style == 10.0) {
        color = band(0.10, 0.17) ? black : white;
    }
    // Anh sang mat phang lam lo khoi va cac chi tiet hinh hoc cua STL.  Muc
    // toi thieu van du sang de khong lam mat cac dai den cua Saturn V.
    vec3 light_direction = normalize(vec3(-0.35, 0.70, 0.60));
    float lambert = max(dot(normalize(v_normal), light_direction), 0.0);
    float illumination = 0.50 + 0.50 * lambert;
    f_color = vec4(color * illumination, 1.0);
}
"""

prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

def discover_stl_paths():
    model_dir = Path(__file__).resolve().parent / "models"
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*.stl"), key=lambda path: path.name.lower())


def build_scene_models(model_mode):
    stl_paths = discover_stl_paths()
    if not stl_paths:
        return [Model(ctx, prog, "models/command moduel.stl")]

    # Rover dung mot file STL hoan chinh. STL goc dung truc Z lam chieu cao,
    # nen xoay sang truc Y cua scene truoc khi render.
    rover_path = next(
        (path for path in stl_paths if path.name.lower() == ROVER_MODEL_FILENAME.lower()),
        None,
    )
    if model_mode == "rover" and rover_path is not None:
        vertices = mesh.Mesh.from_file(str(rover_path)).vectors.reshape(-1, 3)
        rover_scale = ROVER_WORLD_SIZE / max(float((vertices.max(axis=0) - vertices.min(axis=0)).max()), 1e-6)
        rover = Model(
            ctx,
            prog,
            rover_path,
            scale=[rover_scale] * 3,
            local_y_flip=False,
            local_x_rotation=-math.pi / 2,
        )
        rover.is_rover = True
        return [rover]

    roman_path = next((path for path in stl_paths if path.name.lower() == ROMAN_MODEL_FILENAME.lower()), None)
    if model_mode == "roman" and roman_path is not None:
        vertices = mesh.Mesh.from_file(str(roman_path)).vectors.reshape(-1, 3)
        roman_scale = ROMAN_WORLD_SIZE / max(float((vertices.max(axis=0) - vertices.min(axis=0)).max()), 1e-6)
        roman = Model(
            ctx,
            prog,
            roman_path,
            scale=[roman_scale] * 3,
            local_y_flip=False,
            local_x_rotation=-math.pi / 2,
            max_triangles=ROMAN_MAX_TRIANGLES,
        )
        roman.is_roman = True
        return [roman]

    paths_by_role = {path.name.lower(): path for path in stl_paths}
    ordered_paths = [paths_by_role[role] for role in ROCKET_PART_ORDER if role in paths_by_role]

    # "fin (print 4).stl" la mot canh rieng, khong phai than rocket. Tam thoi
    # khong ve no den khi co 4 ban sao + transform rieng de gan dung vi tri.
    # Neu ve no nhu mot stage, no se xuat hien nhu mot manh bi roi.
    if not ordered_paths:
        ordered_paths = stl_paths

    # Do chieu cao STL truoc de tinh scale chung cho toan bo rocket.
    total_native_height = 0.0
    for path in ordered_paths:
        vertices = mesh.Mesh.from_file(str(path)).vectors.reshape(-1, 3)
        total_native_height += float(vertices[:, 1].max() - vertices[:, 1].min())
    shared_scale = ROCKET_WORLD_HEIGHT / max(total_native_height, 1e-6)

    models = []
    next_y = -ROCKET_WORLD_HEIGHT / 2.0
    previous_part_height = None
    for index, path in enumerate(ordered_paths):
        model = Model(ctx, prog, path, scale=[shared_scale] * 3)
        model.rocket_role = path.name.lower()
        # Model da duoc can giua theo bounding box; dat tam cua no ngay sau
        # manh truoc de cac tang cham nhau tren cung truc Y.
        part_height = float((model.mesh.vectors[:, :, 1].max() - model.mesh.vectors[:, :, 1].min()) * shared_scale)
        model.rocket_offset = (
            ROCKET_AXIS_X,
            (next_y + part_height / 2.0) * ROCKET_VERTICAL_FLIP,
            ROCKET_AXIS_Z,
        )
        # Mesh nao cung co khoang trong o hai dau, nen overlap phai lon hon
        # mot chut. Gioi han theo tang nho hon de khong lam mat CM/tower.
        join_overlap = 0.0
        if previous_part_height is not None:
            # Day la cac moi noi cua than chinh. Coupler duoc an vao hai tang
            # lon de S-IC -> S-II -> S-IVB nhin lien khoi khi assembled.
            body_join = model.rocket_role in {
                "s-ic top.stl",
                "stage 1-2 coupler v2.stl",
                "s-ii.stl",
                "s-iv b.stl",
            }
            desired_overlap = 0.24 if body_join else ROCKET_JOIN_OVERLAP
            join_overlap = min(
                desired_overlap,
                part_height * 0.70,
                previous_part_height * 0.70,
            )
        next_y += part_height - join_overlap
        previous_part_height = part_height
        models.append(model)

    # Dat tam cua toan rocket tai goc toa do. Day la pivot DUY NHAT duoc dung
    # luc xoay, nen khi da ghep tat ca cac tang se xoay nhu mot khoi duy nhat.
    actual_min_y = min(
        model.rocket_offset[1] - float((model.mesh.vectors[:, :, 1].max() - model.mesh.vectors[:, :, 1].min()) * shared_scale) / 2.0
        for model in models
    )
    actual_max_y = max(
        model.rocket_offset[1] + float((model.mesh.vectors[:, :, 1].max() - model.mesh.vectors[:, :, 1].min()) * shared_scale) / 2.0
        for model in models
    )
    rocket_pivot_y = (actual_min_y + actual_max_y) / 2.0
    for model in models:
        x, y, z = model.rocket_offset
        trim_y = ROCKET_AXIS_Y_TRIM.get(model.rocket_role, 0.0)
        model.rocket_offset = (x, y - rocket_pivot_y + trim_y, z)

    # Exploded view: 9 slot rieng, co lech tu nhien nhung khong chong nhau.
    for index, model in enumerate(models):
        model.exploded_offset = EXPLODED_SLOTS[index % len(EXPLODED_SLOTS)]

    # STL fin duoc xuat trong cung he toa do voi S-IC bottom, nhung la mot
    # canh don. Tao 4 ban sao quay quanh truc doc. Vi tri lay tu do lech
    # tam giua hai STL goc, nen fin nam dung vung day rocket khi assembled.
    fin_path = paths_by_role.get("fin (print 4).stl")
    sic_bottom = next(
        (model for model in models if model.rocket_role == "s-ic bottom.stl"),
        None,
    )
    if fin_path is not None and sic_bottom is not None:
        native_offset = (mesh.Mesh.from_file(str(fin_path)).vectors.reshape(-1, 3))
        fin_center = (native_offset.min(axis=0) + native_offset.max(axis=0)) / 2.0
        relative_offset = (fin_center - sic_bottom.center) * shared_scale
        # Cung luc lat mesh theo Y o _build_gl_buffers(), nen do lech theo Y
        # so voi S-IC cung phai lat de fin o gan dong co, khong phai tren dau.
        relative_offset[1] *= -1.0
        for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
            fin = Model(
                ctx,
                prog,
                fin_path,
                scale=[shared_scale] * 3,
                local_y_rotation=angle,
            )
            cosine = math.cos(angle)
            sine = math.sin(angle)
            dx, dy, dz = relative_offset
            rotated_offset = np.array(
                [dx * cosine + dz * sine, dy, -dx * sine + dz * cosine],
                dtype=np.float32,
            )
            fin.rocket_role = "fin"
            fin.rocket_offset = tuple(np.asarray(sic_bottom.rocket_offset) + rotated_offset)
            fin.exploded_offset = tuple(np.asarray(sic_bottom.exploded_offset) + rotated_offset)
            models.append(fin)
    return models


def rocket_offset_for(model):
    return getattr(model, "rocket_offset", (0.0, 0.0, 0.0))


scene_models = build_scene_models(ACTIVE_MODEL_MODE)
scene_model = scene_models[0]

def switch_scene_model(model_mode):
    """Nap model duoc xac nhan trong scope chon model."""
    global ACTIVE_MODEL_MODE, scene_models, scene_model
    if model_mode == ACTIVE_MODEL_MODE:
        return
    for model in scene_models:
        for resource in (model.vao, model.vbo, model.ibo):
            resource.release()
    ACTIVE_MODEL_MODE = model_mode
    scene_models = build_scene_models(model_mode)
    scene_model = scene_models[0]


def draw_model_scope(frame, selected_model, hold_seconds):
    """Ve scope de xem model dang duoc chon va tien trinh nam tay xac nhan."""
    height, width = frame.shape[:2]
    center = (width - 105, height - 120)
    radius = 88
    colors = {"rover": (0, 255, 0), "saturn": (0, 220, 255), "roman": (255, 120, 255)}
    color = colors[selected_model]

    cv2.circle(frame, center, radius, (30, 30, 30), -1)
    cv2.circle(frame, center, radius, color, 2)
    cv2.line(frame, (center[0] - radius, center[1]), (center[0] + radius, center[1]), color, 1)
    cv2.line(frame, (center[0], center[1] - radius), (center[0], center[1] + radius), color, 1)
    cv2.putText(frame, "MODEL SCOPE", (center[0] - 44, center[1] - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    cv2.putText(frame, "ROVER", (center[0] - 29, center[1] - 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color if selected_model == "rover" else (170, 170, 170), 1)
    cv2.putText(frame, "SATURN", (center[0] - 34, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color if selected_model == "saturn" else (170, 170, 170), 1)
    cv2.putText(frame, "ROMAN", (center[0] - 30, center[1] + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color if selected_model == "roman" else (170, 170, 170), 1)
    progress = min(hold_seconds / MODEL_SELECTOR_HOLD_SECONDS, 1.0)
    cv2.ellipse(frame, center, (radius - 7, radius - 7), -90, 0, 360 * progress, color, 5)
    cv2.putText(frame, f"{hold_seconds:.1f}/3s", (center[0] - 24, center[1] + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

color_rbo = ctx.renderbuffer((WIDTH, HEIGHT), components=4)
depth_rbo = ctx.depth_renderbuffer((WIDTH, HEIGHT))
fbo = ctx.framebuffer(color_attachments=[color_rbo], depth_attachment=depth_rbo)

proj = pyrr.matrix44.create_perspective_projection_matrix(FOV_DEG, WIDTH / HEIGHT, NEAR, FAR)
view = pyrr.matrix44.create_look_at(eye=[0, 0, EYE_Z], target=[0, 0, 0], up=[0, 1, 0])

_half_h_world = math.tan(math.radians(FOV_DEG / 2)) * EYE_Z
WORLD_PER_PX = (2 * _half_h_world) / HEIGHT


def world_to_screen(x, y):
    return WIDTH / 2 + x / WORLD_PER_PX, HEIGHT / 2 - y / WORLD_PER_PX


def render_cube(angle_x, angle_y, obj_x, obj_y, obj_z, assembly=1.0):
    fbo.use()
    ctx.viewport = (0, 0, WIDTH, HEIGHT)
    fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

    # xoay quanh truc X (nghieng len/xuong) roi den truc Y (quay trai/phai)
    rot_x = pyrr.matrix44.create_from_x_rotation(angle_x)
    rot_y = pyrr.matrix44.create_from_y_rotation(angle_y)
    rotation = pyrr.matrix44.multiply(rot_x, rot_y)
    translation = pyrr.matrix44.create_from_translation([obj_x, obj_y, obj_z])
    # Transform chung cua ca rocket: 1 vi tri va 1 truc/pivot xoay.
    rocket_group_matrix = pyrr.matrix44.multiply(rotation, translation)

    mv = pyrr.matrix44.multiply(rocket_group_matrix, view)
    mvp = pyrr.matrix44.multiply(mv, proj)

    prog["mvp"].write(mvp.astype("f4").tobytes())
    if scene_models:
        for model in scene_models:
            assembled = rocket_offset_for(model)
            exploded = getattr(model, "exploded_offset", assembled)
            part_offset_x, part_offset_y, part_offset_z = tuple(
                exploded_value + (assembled_value - exploded_value) * assembly
                for exploded_value, assembled_value in zip(exploded, assembled)
            )
            part_translation = pyrr.matrix44.create_from_translation([
                part_offset_x,
                part_offset_y,
                part_offset_z,
            ])
            # Pyrr/OpenGL o day dung row-vector convention: offset cua part
            # phai duoc ap truoc, roi ca rocket moi quay va dich chuyen. Neu
            # de nguoc thu tu, moi part bi dich sau khi xoay va trong nhu quay
            # quanh mot truc rieng.
            model_matrix = pyrr.matrix44.multiply(part_translation, rocket_group_matrix)
            mv = pyrr.matrix44.multiply(model_matrix, view)
            model_mvp = pyrr.matrix44.multiply(mv, proj)
            model.draw(model_mvp)
    raw = fbo.read(components=4, dtype="f1")
    img = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 4)).copy()
    return np.flipud(img)


def composite(frame_bgr, render_rgba):
    if render_rgba.shape[:2] != frame_bgr.shape[:2]:
        render_rgba = cv2.resize(render_rgba, (frame_bgr.shape[1], frame_bgr.shape[0]))
    rgb = render_rgba[:, :, :3][:, :, ::-1]
    alpha = render_rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = rgb.astype(np.float32) * alpha + frame_bgr.astype(np.float32) * (1 - alpha)
    return out.astype(np.uint8)


def normalized_depth_delta(new_size, previous_size):
    if previous_size <= 0:
        return 0.0
    ratio = (new_size - previous_size) / previous_size
    return clamp(ratio * Z_SENSITIVITY, -0.35, 0.35)


def screen_delta_to_world(dx, dy):
    return dx * WORLD_PER_PX, -dy * WORLD_PER_PX


def update_grab_target(target_x, target_y, target_z, grab_px, last_grab_px):
    dx = apply_dead_zone(grab_px[0] - last_grab_px[0], MOVE_DEAD_ZONE_PX)
    dy = apply_dead_zone(grab_px[1] - last_grab_px[1], MOVE_DEAD_ZONE_PX)
    dz = normalized_depth_delta(grab_px[2], last_grab_px[2])
    dz = apply_dead_zone(dz, DEPTH_DEAD_ZONE)

    world_dx, world_dy = screen_delta_to_world(dx, dy)
    target_x += world_dx
    target_y += world_dy
    target_z = clamp(target_z + dz, Z_MIN, Z_MAX)
    return target_x, target_y, target_z


def update_rotate_target(target_angle_x, target_angle_y, rotate_px, last_rotate_px):
    dx = apply_dead_zone(rotate_px[0] - last_rotate_px[0], ROTATE_DEAD_ZONE_PX)
    dy = apply_dead_zone(rotate_px[1] - last_rotate_px[1], ROTATE_DEAD_ZONE_PX)
    target_angle_y += dx * ROTATE_SENSITIVITY
    target_angle_x += -dy * ROTATE_SENSITIVITY
    return target_angle_x, target_angle_y


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cap = open_camera()
    if cap is None or not cap.isOpened():
        print(
            "Cannot open webcam.\n"
            "Try closing other camera apps, check camera permissions, or use a different device index."
        )
        return

    angle_x = 0.0
    angle_y = 0.0
    obj_x, obj_y, obj_z = 0.0, 0.0, 0.0
    is_grabbing = False
    last_grab_px = None
    last_rotate_px = None

    grab_state = GestureState(pinch_ratio, pinch_point_px, PINCH_ENTER, PINCH_EXIT)
    rotate_state = GestureState(three_finger_pinch_ratio, palm_center_px, THREE_PINCH_ENTER, THREE_PINCH_EXIT)
    target_x, target_y, target_z = obj_x, obj_y, obj_z
    target_angle_x, target_angle_y = angle_x, angle_y
    # Scope chon model dung fist giu 3 giay, nen Saturn V luon bat dau o trang
    # thai da ghep de fist khong bi trung chuc nang assemble cu.
    assembly = 1.0
    rocket_is_assembled = True
    selected_model = ACTIVE_MODEL_MODE
    last_selector_angle = None
    selector_rotation = 0.0
    fist_hold_started = None
    fist_selection_confirmed = False
    two_open_frame_count = 0
    grab_filter = VecEuroFilter(3, min_cutoff=0.6, beta=0.15)
    rotate_filter = VecEuroFilter(2, min_cutoff=0.6, beta=0.15)
    missed_frame_count = 0
    MAX_MISSED_FRAMES = 20
    reopen_attempts = 0
    MAX_REOPEN_ATTEMPTS = 3

    while True:
        is_rover_scene = bool(scene_models and getattr(scene_models[0], "is_rover", False))
        is_roman_scene = bool(scene_models and getattr(scene_models[0], "is_roman", False))
        is_saturn_scene = not is_rover_scene and not is_roman_scene
        ret, frame = cap.read()
        if not ret or frame is None:
            missed_frame_count += 1
            if missed_frame_count >= MAX_MISSED_FRAMES:
                cap.release()
                reopen_attempts += 1
                if reopen_attempts > MAX_REOPEN_ATTEMPTS:
                    print("Cannot read frame. Closing after repeated failures.")
                    break
                print(f"Camera lost: attempting reopen ({reopen_attempts}/{MAX_REOPEN_ATTEMPTS})...")
                cap = open_camera()
                if cap is None or not cap.isOpened():
                    print("Reopen failed.")
                    break
                missed_frame_count = 0
                continue
            continue
        missed_frame_count = 0

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_landmarks = None
        all_landmark_lists = []

        if results.multi_hand_landmarks:
            for hand_index, detected_hand in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, detected_hand, mp_hands.HAND_CONNECTIONS)
                all_landmark_lists.append(detected_hand.landmark)

        # Khong phu thuoc nhan Left/Right cua webcam (de bi dao sau khi mirror).
        # Chum cai-tro = grab; chum ca cai-tro-giua = rotate, o bat ky tay nao.
        grab_landmarks = gesture_hand(all_landmark_lists, pinch_ratio, w, h)
        rotate_landmarks = gesture_hand(all_landmark_lists, three_finger_pinch_ratio, w, h)

        now = time.time()
        grab_px = grab_state.update(grab_landmarks, w, h, now=now)
        rotate_px = rotate_state.update(rotate_landmarks, w, h, now=now)

        if grab_px is not None:
            grab_px = grab_filter.filter(grab_px, now)
        elif not (grab_state.active and grab_state.missing_since is not None and now - grab_state.missing_since < grab_state.grace_period):
            grab_filter.reset()

        if rotate_px is not None:
            rotate_px = rotate_filter.filter(rotate_px, now)
        elif not (rotate_state.active and rotate_state.missing_since is not None and now - rotate_state.missing_since < rotate_state.grace_period):
            rotate_filter.reset()

        # ==== Grab / di chuyen ====
        obj_screen = world_to_screen(obj_x, obj_y)

        hand_present = grab_landmarks is not None
        is_pinching = grab_state.active
        is_rotating = rotate_state.active
        fist_detected = any(is_fist_hand(points, w, h) for points in all_landmark_lists)
        is_fist = fist_detected

        # Xoe tay va xoay co tay trai/phai de doi muc dang sang trong scope.
        selector_hand = next((points for points in all_landmark_lists if is_open_hand(points)), None)
        if selector_hand is not None and not is_pinching and not is_rotating and not is_fist:
            current_selector_angle = palm_rotation_angle(selector_hand, w, h)
            if last_selector_angle is not None:
                selector_rotation += wrapped_angle_delta(current_selector_angle, last_selector_angle)
                if abs(selector_rotation) >= MODEL_SELECTOR_STEP_RAD:
                    current_index = MODEL_CHOICES.index(selected_model)
                    direction = 1 if selector_rotation > 0 else -1
                    selected_model = MODEL_CHOICES[(current_index + direction) % len(MODEL_CHOICES)]
                    selector_rotation = 0.0
            last_selector_angle = current_selector_angle
        else:
            last_selector_angle = None
            selector_rotation = 0.0

        # Nam tay lien tuc 3 giay de xac nhan model dang sang.
        if is_fist:
            if fist_hold_started is None:
                fist_hold_started = now
            fist_hold_seconds = now - fist_hold_started
            if fist_hold_seconds >= MODEL_SELECTOR_HOLD_SECONDS and not fist_selection_confirmed:
                switch_scene_model(selected_model)
                rocket_is_assembled = True
                fist_selection_confirmed = True
        else:
            # Nam tay nhanh la lenh ghep Saturn V; giu du 3 giay van danh
            # rieng cho viec xac nhan model trong scope.
            if is_saturn_scene and fist_hold_started is not None and not fist_selection_confirmed:
                rocket_is_assembled = True
            fist_hold_started = None
            fist_hold_seconds = 0.0
            fist_selection_confirmed = False

        # Hai ban tay xoe lien tuc se tach cac tang Saturn V. Khong ap dung
        # cho rover/Roman de tranh tac dong ngoai y muon.
        two_hands_open = (
            len(all_landmark_lists) >= 2
            and all(is_open_hand(points) for points in all_landmark_lists[:2])
        )
        two_open_frame_count = two_open_frame_count + 1 if two_hands_open else 0
        if is_saturn_scene and two_open_frame_count >= OPEN_HAND_DEBOUNCE_FRAMES:
            rocket_is_assembled = False
        obj_screen = world_to_screen(obj_x, obj_y)
        hand_distance = None

        if hand_present and grab_px is not None:
            hand_distance = math.hypot(grab_px[0] - obj_screen[0], grab_px[1] - obj_screen[1])

        # ==== Grab Mode ====
        # Rotate uu tien cao nhat. Truoc day fist duoc kiem tra truoc, nen
        # overlay bao ROTATE nhung nhanh xoay lai bi chan.
        if is_rotating and rotate_px is not None:
            is_grabbing = False
            last_grab_px = None
            if last_rotate_px is None:
                last_rotate_px = rotate_px
            else:
                target_angle_x, target_angle_y = update_rotate_target(
                    target_angle_x, target_angle_y, rotate_px, last_rotate_px
                )
                last_rotate_px = rotate_px
        elif is_fist:
            is_grabbing = False
            last_grab_px = None
            last_rotate_px = None
        else:
            last_rotate_px = None

        # Pinch van duoc keo vat ngay ca khi fist-state vua chuyen doi. Neu
        # khong, nguoi dung thay PINCH tren man hinh nhung vat khong di chuyen.
        if is_pinching and not is_rotating and grab_px is not None:
            if not is_grabbing:
                if hand_distance is not None and hand_distance < GRAB_RADIUS_PX:
                    is_grabbing = True
                    last_grab_px = grab_px
                    target_x, target_y, target_z = obj_x, obj_y, obj_z
            elif last_grab_px is not None:
                target_x, target_y, target_z = update_grab_target(
                    target_x, target_y, target_z, grab_px, last_grab_px
                )
                last_grab_px = grab_px
        else:
            if not is_rotating:
                is_grabbing = False
                last_grab_px = None

        obj_x = smooth_value(obj_x, target_x, 0.2)
        obj_y = smooth_value(obj_y, target_y, 0.2)
        obj_z = clamp(smooth_value(obj_z, target_z, 0.2), Z_MIN, Z_MAX)

        angle_x = smooth_value(angle_x, target_angle_x, 0.15)
        angle_y = smooth_value(angle_y, target_angle_y, 0.15)
        assembly = smooth_value(assembly, 1.0 if rocket_is_assembled else 0.0, 0.12)

        # ==== Render + composite ====
        render_img = render_cube(angle_x, angle_y, obj_x, obj_y, obj_z, assembly)
        composited = composite(frame, render_img)

        # Debug overlay
        grab_status = "GRAB" if is_grabbing else ("PINCH" if grab_state.active else "-")
        rotate_status = "ROTATE" if rotate_state.active else "-"
        assemble_status = "ASSEMBLED" if rocket_is_assembled else "EXPLODED"

        grab_color = (
            (0, 0, 255)
            if is_grabbing
            else ((0, 165, 255) if grab_state.active else (0, 255, 0))
        )

        cv2.putText(
            composited,
            f"{GRAB_HAND_LABEL} hand (grab): {grab_status}  z={obj_z:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            grab_color,
            2,
        )

        cv2.putText(
            composited,
            f"{ROTATE_HAND_LABEL} hand (rotate): {rotate_status}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

        instruction = (
            "Curiosity Rover | Pinch = move | Three-finger pinch = rotate"
            if is_rover_scene
            else (
                "Nancy Grace Roman Telescope | Pinch = move | Three-finger pinch = rotate"
                if scene_models and getattr(scene_models[0], "is_roman", False)
                else "Saturn V | Fist short = assemble | 2 open hands = explode"
            )
        )
        cv2.putText(
            composited,
            instruction,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255) if is_rover_scene else ((255, 255, 255) if rocket_is_assembled else (180, 180, 180)),
            2,
        )
        cv2.putText(
            composited,
            "Open palm + rotate = choose model | Hold fist 3s = confirm | Q = quit",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        draw_model_scope(composited, selected_model, fist_hold_seconds)

        cv2.imshow("Two-hand Grab + Rotate", composited)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
