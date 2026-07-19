"""Camera model for 3D-aware court projection.

Uses cv2.solvePnP to estimate camera pose from known 3D court points
(ground keypoints + net post tops at known height). This gives a proper
camera projection matrix that handles perspective correctly, unlike
flat homography which only works on the ground plane.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

# Court dimensions (meters)
COURT_WIDTH = 10.0
COURT_LENGTH = 20.0
NET_Y = 10.0
NET_HEIGHT = 0.88  # net height at center
NET_POST_HEIGHT = 0.92  # net height at posts
SERVICE_NEAR_Y = 6.95
SERVICE_FAR_Y = 13.05
SERVICE_CENTER_X = 5.0

# 12 ground keypoints at Z=0 (same order as court_calibration.py)
GROUND_KEYPOINTS_3D = np.array([
    [0, 0, 0],                                # k1  near-left baseline
    [COURT_WIDTH, 0, 0],                       # k2  near-right baseline
    [0, SERVICE_NEAR_Y, 0],                    # k3  near-left service
    [SERVICE_CENTER_X, SERVICE_NEAR_Y, 0],     # k4  near-center service
    [COURT_WIDTH, SERVICE_NEAR_Y, 0],          # k5  near-right service
    [0, NET_Y, 0],                             # k6  net-left (ground)
    [COURT_WIDTH, NET_Y, 0],                   # k7  net-right (ground)
    [0, SERVICE_FAR_Y, 0],                     # k8  far-left service
    [SERVICE_CENTER_X, SERVICE_FAR_Y, 0],      # k9  far-center service
    [COURT_WIDTH, SERVICE_FAR_Y, 0],           # k10 far-right service
    [0, COURT_LENGTH, 0],                      # k11 far-left baseline
    [COURT_WIDTH, COURT_LENGTH, 0],            # k12 far-right baseline
], dtype=np.float64)

# Net post tops at known height (Z = NET_POST_HEIGHT)
NET_TOP_KEYPOINTS_3D = np.array([
    [0, NET_Y, NET_POST_HEIGHT],               # net-left top
    [COURT_WIDTH, NET_Y, NET_POST_HEIGHT],     # net-right top
], dtype=np.float64)


class CameraModel:
    """3D camera model estimated from court keypoints.

    Provides:
    - project_to_ground(px, py) → (court_x, court_y) for objects on the ground
    - project_to_height(px, py, height) → (court_x, court_y) for objects at known height
    - court_to_pixel(cx, cy, cz=0) → (px, py) for drawing overlays
    """

    def __init__(self):
        self.camera_matrix: Optional[np.ndarray] = None  # 3x3 intrinsic matrix
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvec: Optional[np.ndarray] = None  # rotation vector
        self.tvec: Optional[np.ndarray] = None  # translation vector
        self.homography: Optional[np.ndarray] = None  # fallback 2D homography
        self.inverse_homography: Optional[np.ndarray] = None

    @classmethod
    def from_parameters(cls, camera_matrix, rotation_world_to_camera,
                        translation_world_to_camera, dist_coeffs=None):
        """Build a camera model from known pinhole intrinsics/extrinsics.

        Dataset and installed smart-court cameras often already provide a
        calibrated ``K, R, t``. Re-solving PnP from clicked court landmarks
        would discard that information and add avoidable error.
        """
        camera = cls()
        camera.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        rotation = np.asarray(rotation_world_to_camera, dtype=np.float64).reshape(3, 3)
        camera.rvec, _ = cv2.Rodrigues(rotation)
        camera.tvec = np.asarray(translation_world_to_camera, dtype=np.float64).reshape(3, 1)
        if dist_coeffs is None:
            camera.dist_coeffs = np.zeros(5, dtype=np.float64)
        else:
            camera.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        return camera

    def calibrate(self, keypoints_2d: List[List[float]],
                  net_top_2d: Optional[List[List[float]]] = None,
                  image_width: int = 1280, image_height: int = 720) -> None:
        """Calibrate camera from 2D keypoint observations.

        Args:
            keypoints_2d: 4-12 ground keypoint pixel positions
            net_top_2d: Optional 2 net post TOP pixel positions (for 3D calibration)
            image_width, image_height: frame dimensions for initial intrinsics estimate
        """
        n_ground = min(len(keypoints_2d), 12)

        # Build 3D-2D correspondences
        pts_3d = GROUND_KEYPOINTS_3D[:n_ground].copy()
        pts_2d = np.array(keypoints_2d[:n_ground], dtype=np.float64)

        # Add net top points if provided (gives Z-axis constraint)
        if net_top_2d is not None and len(net_top_2d) == 2:
            pts_3d = np.vstack([pts_3d, NET_TOP_KEYPOINTS_3D])
            pts_2d = np.vstack([pts_2d, np.array(net_top_2d, dtype=np.float64)])

        # Always compute ground-plane homography as fallback
        if n_ground >= 4:
            from cv.court_calibration import KEYPOINT_COURT_COORDS_12
            src = np.array(keypoints_2d[:n_ground], dtype=np.float32)
            dst = np.array(KEYPOINT_COURT_COORDS_12[:n_ground], dtype=np.float32)
            self.homography, _ = cv2.findHomography(src, dst)
            # Use the algebraic inverse of the accepted mapping. Fitting a
            # second homography independently to noisy points does not produce
            # an inverse and breaks court→pixel→court round trips.
            self.inverse_homography = np.linalg.inv(self.homography)

        # Try full camera calibration if we have enough points
        if len(pts_3d) >= 6:
            # Initial camera matrix estimate
            fx = image_width  # rough focal length
            self.camera_matrix = np.array([
                [fx, 0, image_width / 2.0],
                [0, fx, image_height / 2.0],
                [0, 0, 1],
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros(4, dtype=np.float64)

            # Solve for camera pose
            success, rvec, tvec = cv2.solvePnP(
                pts_3d, pts_2d,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success:
                self.rvec = rvec
                self.tvec = tvec

                # Refine with Levenberg-Marquardt
                self.rvec, self.tvec = cv2.solvePnPRefineLM(
                    pts_3d, pts_2d,
                    self.camera_matrix, self.dist_coeffs,
                    self.rvec, self.tvec,
                )

    def project_to_ground(self, px: float, py: float) -> Tuple[float, float]:
        """Project pixel to ground plane (Z=0). Best for player feet and ball bounces.

        Prefer the directly-fitted ground homography: it maps the ground plane
        exactly, whereas the PnP pose (decomposed with an estimated focal length)
        carries reprojection error. Use the PnP ray-plane intersect only when no
        homography is available.
        """
        if self.homography is not None:
            point = np.array([[[px, py]]], dtype=np.float32)
            result = cv2.perspectiveTransform(point, self.homography)
            return float(result[0][0][0]), float(result[0][0][1])
        if self.rvec is not None:
            return self._ray_plane_intersect(px, py, z_plane=0.0)
        raise RuntimeError("Camera not calibrated")

    def project_to_height(self, px: float, py: float, height: float) -> Tuple[float, float]:
        """Project pixel to a horizontal plane at given height. For ball in air."""
        if self.rvec is not None:
            return self._ray_plane_intersect(px, py, z_plane=height)
        # Fallback: just use ground projection
        return self.project_to_ground(px, py)

    def court_to_pixel(self, cx: float, cy: float, cz: float = 0.0) -> Tuple[float, float]:
        """Project 3D court point to pixel coordinates."""
        # The ground homography is fitted directly to every ground keypoint and
        # must be the inverse of project_to_ground.  A PnP pose inferred from
        # coplanar points and guessed intrinsics is less accurate and is only
        # needed once height is non-zero.
        if abs(cz) < 1e-9 and self.inverse_homography is not None:
            point = np.array([[[cx, cy]]], dtype=np.float32)
            result = cv2.perspectiveTransform(point, self.inverse_homography)
            return float(result[0][0][0]), float(result[0][0][1])
        if self.rvec is not None:
            pt_3d = np.array([[cx, cy, cz]], dtype=np.float64)
            pts_2d, _ = cv2.projectPoints(pt_3d, self.rvec, self.tvec,
                                           self.camera_matrix, self.dist_coeffs)
            return float(pts_2d[0][0][0]), float(pts_2d[0][0][1])
        elif self.inverse_homography is not None:
            point = np.array([[[cx, cy]]], dtype=np.float32)
            result = cv2.perspectiveTransform(point, self.inverse_homography)
            return float(result[0][0][0]), float(result[0][0][1])
        raise RuntimeError("Camera not calibrated")

    def projection_matrix(self) -> Optional[np.ndarray]:
        """Return the 3x4 camera projection matrix P = K [R | t].

        Maps a homogeneous 3D world point to homogeneous pixels. Required for
        multi-camera triangulation. None if the 3D pose was not solved.
        """
        if self.rvec is None or self.camera_matrix is None:
            return None
        R, _ = cv2.Rodrigues(self.rvec)
        Rt = np.hstack([R, self.tvec.reshape(3, 1)])
        return self.camera_matrix @ Rt

    def pixel_ray(self, px: float, py: float):
        """Return ``(camera_origin, world_direction)`` for an image pixel."""
        if self.camera_matrix is None or self.rvec is None or self.tvec is None:
            raise RuntimeError("3D camera model is not calibrated")
        R, _ = cv2.Rodrigues(self.rvec)
        origin = (-R.T @ self.tvec.reshape(3, 1)).reshape(3)
        pixel = np.array([px, py, 1.0], dtype=np.float64)
        direction_camera = np.linalg.inv(self.camera_matrix) @ pixel
        direction_world = R.T @ direction_camera
        direction_world /= np.linalg.norm(direction_world)
        return tuple(float(v) for v in origin), tuple(float(v) for v in direction_world)

    def _ray_plane_intersect(self, px: float, py: float, z_plane: float) -> Tuple[float, float]:
        """Cast a ray from pixel through camera and intersect with Z=z_plane."""
        # Get rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        # Camera position in world coords
        cam_pos = -R.T @ self.tvec.flatten()

        # Ray direction in camera coords
        px_normalized = np.array([
            (px - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0],
            (py - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1],
            1.0,
        ])
        # Transform to world coords
        ray_dir = R.T @ px_normalized

        # Intersect with plane Z = z_plane
        if abs(ray_dir[2]) < 1e-10:
            # Ray parallel to plane — fallback to homography
            return self.project_to_ground(px, py) if z_plane == 0 else (0, 0)

        t = (z_plane - cam_pos[2]) / ray_dir[2]
        world_point = cam_pos + t * ray_dir

        return float(world_point[0]), float(world_point[1])

    def estimate_height_from_size(self, bbox_height_px: float,
                                  foot_court_y: float,
                                  known_height_m: float = 1.8) -> float:
        """Estimate real-world height of an object from its pixel size.

        Uses the camera model to determine how many meters one pixel
        represents at the object's court position.
        """
        if self.rvec is None:
            return 0.0

        # Project foot position and a point 1m above it
        foot_px = self.court_to_pixel(5.0, foot_court_y, 0.0)
        head_px = self.court_to_pixel(5.0, foot_court_y, known_height_m)

        # Pixels per meter at this depth
        px_per_m = abs(foot_px[1] - head_px[1]) / known_height_m
        if px_per_m < 1:
            return 0.0

        return bbox_height_px / px_per_m

    # --- Adapter methods (CourtCalibration interface) ---

    def pixel_to_court(self, px: float, py: float):
        """Adapter: matches CourtCalibration.pixel_to_court signature."""
        return self.project_to_ground(px, py)

    def court_to_pixel_2d(self, cx: float, cy: float):
        """Adapter: 2D court coords to pixel (ground plane)."""
        return self.court_to_pixel(cx, cy, 0.0)

    def is_in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x <= COURT_WIDTH and 0.0 <= y <= COURT_LENGTH

    def get_court_side(self, x: float, y: float) -> str:
        return "near" if y < NET_Y else "far"

    def is_in_service_box(self, x: float, y: float, box: str) -> bool:
        boxes = {
            "near_left": (0, 5, 6.95, 10),
            "near_right": (5, 10, 6.95, 10),
            "far_left": (0, 5, 10, 13.05),
            "far_right": (5, 10, 10, 13.05),
        }
        if box not in boxes:
            return False
        x1, x2, y1, y2 = boxes[box]
        return x1 <= x <= x2 and y1 <= y <= y2

    def has_3d(self) -> bool:
        """Whether full 3D camera model is available (vs fallback homography)."""
        return self.rvec is not None

    def to_dict(self) -> dict:
        result = {}
        if self.camera_matrix is not None:
            result["camera_matrix"] = self.camera_matrix.tolist()
        if self.dist_coeffs is not None:
            result["dist_coeffs"] = self.dist_coeffs.tolist()
        if self.rvec is not None:
            result["rvec"] = self.rvec.tolist()
        if self.tvec is not None:
            result["tvec"] = self.tvec.tolist()
        if self.homography is not None:
            result["homography"] = self.homography.tolist()
        if self.inverse_homography is not None:
            result["inverse_homography"] = self.inverse_homography.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "CameraModel":
        cam = cls()
        if "camera_matrix" in data:
            cam.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        if "dist_coeffs" in data:
            cam.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
        if "rvec" in data:
            cam.rvec = np.array(data["rvec"], dtype=np.float64)
        if "tvec" in data:
            cam.tvec = np.array(data["tvec"], dtype=np.float64)
        if "homography" in data:
            cam.homography = np.array(data["homography"], dtype=np.float64)
        if "inverse_homography" in data:
            cam.inverse_homography = np.array(data["inverse_homography"], dtype=np.float64)
        return cam
