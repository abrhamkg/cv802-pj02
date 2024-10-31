from pathlib import Path

import numpy as np
import open3d as o3d
import os
import os.path as osp
import glob
import pycolmap
import pdb
import collections
from utils.thread_utils import run_on_thread
import struct
import time

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

VOCAB_PATH = 'modules/colmap/vocab_tree_flickr100K_words32K.bin'



CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)



def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D

def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):

    cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
    images = read_images_binary(os.path.join(path, "images" + ext))
    points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)

    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


class ColmapAPI:
    def __init__(
        self,
        gpu_index,
        camera_model,
        matcher,
    ):
        self._data_path = None
        self._pcd = None
        self._thread = None
        self._active_camera_name = None
        self._cameras = dict()
        self._vis = None

        self._gpu_index = gpu_index
        self._camera_model = camera_model
        self._matcher = matcher
        if self._matcher not in ['exhaustive_matcher', 'vocab_tree_matcher', 'sequential_matcher']:
            raise ValueError(f'Only support exhaustive_matcher and vocab_tree_matcher, got {self._matcher}')

    @property
    def data_path(self):
        if self._data_path is None:
            raise ValueError(f'Data path was not set')
        return self._data_path

    @data_path.setter
    def data_path(self, new_data_path):
        self._data_path = new_data_path

    @property
    def image_dir(self):
        return osp.join(self.data_path, 'images')

    @property
    def database_path(self):
        return osp.join(self.data_path, 'colmap/database.db')

    @property
    def sparse_dir(self):
        return osp.join(self.data_path, 'colmap/sparse')

    @property
    def num_cameras(self):
        return len(self._cameras)

    @property
    def camera_names(self):
        return list(self._cameras.keys())

    @property
    def pcd(self):
        if self._pcd is None:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        return self._pcd

    @property
    def activate_camera_name(self):
        if len(self._cameras) == 0:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        return self._active_camera_name

    @activate_camera_name.setter
    def activate_camera_name(self, new_value):
        if len(self._cameras) == 0:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        self._active_camera_name = new_value

    @property
    def camera_model(self):
        return self._camera_model

    @camera_model.setter
    def camera_model(self, new_value):
        self._camera_model = new_value 

    @property
    def matcher(self):
        return self._matcher

    @matcher.setter
    def matcher(self, new_value):
        self._matcher = new_value

    def check_colmap_folder_valid(self):
        database_path = self.database_path
        image_dir = self.image_dir
        sparse_dir = self.sparse_dir

        print('Database file:', database_path)
        print('Image path:', image_dir)
        print('Bundle adjustment path:', sparse_dir)

        is_valid = \
            osp.isfile(database_path) and \
            osp.isdir(image_dir) and \
            osp.isdir(sparse_dir)

        return is_valid

    @run_on_thread
    def _estimate_cameras(self, recompute):
        ''' Assignment 1

        In this assignment, you need to compute two things:
            pcd: A colored point cloud represented using open3d.geometry.PointCloud
            cameras: A dictionary of the following format:
                {
                    camera_name_01 [str]: {
                        'extrinsics': [rotation [Matrix 3x3], translation [Vector 3]]
                        'intrinsics': {
                            'width': int
                            'height': int
                            'fx': float
                            'fy': float
                            'cx': float
                            'cy': float
                        }
                    }
                    ...
                }

            You can check the extract_camera_parameters method to understand how the cameras are used.
        '''

        ## Insert your code below -------------------------------------------------------------------

        #Verify Colmap Path
        colmap_dir = os.path.join(self.data_path, 'colmap')
        if not os.path.exists(colmap_dir):
            os.makedirs(colmap_dir)

        print("-------------------", self.data_path)
        images = Path(os.path.join(self.data_path, 'train'))
        outputs = Path(os.path.join(self.data_path, 'output'))
        sfm_pairs = outputs / 'pairs-sfm.txt'
        loc_pairs = outputs / 'pairs-loc.txt'
        sfm_dir = outputs / 'sfm'
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'
        if recompute:
            feature_conf = extract_features.confs['disk']
            matcher_conf = match_features.confs['NN-ratio']
            matcher_conf = match_features.confs['disk+lightglue']

            references = [str(p.relative_to(images)) for p in (images / 'mapping/').iterdir()]
            extract_features.main(feature_conf, images, image_list=references, feature_path=features)
            pairs_from_exhaustive.main(sfm_pairs, image_list=references)
            match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches);

            model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)

        #Load saved bin model
        cameras_bin, images_bin, points3D_bin = read_model(
            sfm_dir,
            ext=".bin")
        pdb.set_trace()
        #Projection Error calculation
        proj_error=0
        for e in points3D_bin.keys():
            proj_error+=points3D_bin[e].error.item()

        proj_error/=len(points3D_bin.keys())
        print("Mean Reprojection Error",proj_error)

        #Get 3D points
        points_xyz=[]
        points_rgb=[]
        for pts in points3D_bin.keys():
            points_xyz.append(points3D_bin[pts].xyz)
            points_rgb.append(points3D_bin[pts].rgb/255)

        #Get Image and Camera Data
        camera={}
        for img in images_bin.keys():
            img_cam_id = images_bin[img].camera_id

            #Get instrinsic parameters from cameras
            fx=cameras_bin[img_cam_id].params[0]
            fy=cameras_bin[img_cam_id].params[1]

            #Check camera type
            if self.camera_model=="SIMPLE_PINHOLE" or self.camera_model=="PINHOLE":
                cx=0
                cy=0
            else:
                cx=cameras_bin[img_cam_id].params[2]
                cy=cameras_bin[img_cam_id].params[3]
            width=cameras_bin[img_cam_id].width
            height=cameras_bin[img_cam_id].height
            #Get extrinsic parameters from cameras
            rot=qvec2rotmat(images_bin[img].qvec)
            trans=(images_bin[img].tvec)

            #Final Matrix
            camera[str(img)]={
                    'extrinsic': [rot,trans],
                    'intrinsic': {
                        'width': width,
                        'height': height,
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy,
                    }}

        # Add points
        pcd = o3d.geometry.PointCloud()
        pcd.points.extend(points_xyz)
        pcd.colors.extend(points_rgb)
        
        # Add cameras
        colmap_cameras = camera
        o3d.io.write_point_cloud("point_cloud_for_meshing.ply", pcd, write_ascii=True)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)

        o3d.io.write_point_cloud("point_cloud_point2mesh.xyz", cl, write_ascii=True)
        ####### End of your code #####################

        self._pcd = pcd
        self._cameras = colmap_cameras
        self.activate_camera_name = self.camera_names[0]

    @staticmethod
    def _list_images_in_folder(directory):
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg'}
        files = sorted(glob.glob(osp.join(directory, '*')))
        files = list(filter(lambda x: osp.splitext(x)[1].lower() in image_extensions, files)) 
        return files

    def estimate_done(self):
        return not self._thread.is_alive()

    def estimate_cameras(self, recompute=True):
        self._thread = self._estimate_cameras(recompute)

    def extract_camera_parameters(self, camera_name):
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self._cameras[camera_name]['intrinsic']['width'],
            self._cameras[camera_name]['intrinsic']['height'],
            self._cameras[camera_name]['intrinsic']['fx'],
            self._cameras[camera_name]['intrinsic']['fy'],
            self._cameras[camera_name]['intrinsic']['cx'],
            self._cameras[camera_name]['intrinsic']['cy'],
        )

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = self._cameras[camera_name]['extrinsic'][0]
        extrinsics[:3, 3] = self._cameras[camera_name]['extrinsic'][1]
        extrinsics = extrinsics

        return intrinsics, extrinsics
