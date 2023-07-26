import glob
import json
import os
import pickle

import numpy as np
import torch
from constants import (
    AUGMENTED_VERTICES_INDEX_DICT,
    AUGMENTED_VERTICES_NAMES,
    COCO_VERTICES_NAME,
    IMAGES_ROOT_PATH,
    SEQ_NAMES,
    SET,
    SMPLX_MODEL_DIR,
)
from smplx import SMPLX
from tqdm import tqdm
from utils import CalibratedCamera


class DatasetGenerator:
    def __init__(
        self,
        output_path: str = "infinity_dataset_combined",
        sample_rate: int = 6,
    ):
        self.img_width = 4112
        self.img_height = 3008
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.data_dict = {
            "infos": {},
            "images": [],
            "annotations": [],
            "categories": [],
        }

        self.data_dict["categories"] = [
            {
                "id": 0,
                "augmented_keypoints": AUGMENTED_VERTICES_NAMES,
                "coco_keypoints": COCO_VERTICES_NAME,
            }
        ]
        self.total_source_images = 0
        self.total_error_reconstruction = 0

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_bbox(self, vertices):
        x_img, y_img = vertices[:, 0], vertices[:, 1]
        xmin = min(x_img)
        ymin = min(y_img)
        xmax = max(x_img)
        ymax = max(y_img)

        x_center = (xmin + xmax) / 2.0
        width = xmax - xmin
        xmin = x_center - 0.5 * width  # * 1.2
        xmax = x_center + 0.5 * width  # * 1.2

        y_center = (ymin + ymax) / 2.0
        height = ymax - ymin
        ymin = y_center - 0.5 * height  # * 1.2
        ymax = y_center + 0.5 * height  # * 1.2

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(self.img_width, xmax)
        ymax = min(self.img_height, ymax)

        bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(int)

        return bbox

    def generate_annotation_dict(self):
        annotation_dict = {}
        annotation_dict["image_id"] = len(self.data_dict["images"])
        annotation_dict["id"] = annotation_dict["image_id"]
        annotation_dict["category_id"] = 0
        annotation_dict["iscrowd"] = 0

        return annotation_dict

    def get_grountruth_landmarks(
        self,
        body_model,
        set: str,
        scene_name: str,
        seq_name: str,
        camera_id: int,
        frame_id: int,
        sub_id,
    ):
        smplx_params_fn = os.path.join(
            "resource/bodies", set, seq_name, f"{frame_id:05d}", f"{sub_id}.pkl"
        )
        body_params = pickle.load(open(smplx_params_fn, "rb"))
        body_params = {k: torch.from_numpy(v) for k, v in body_params.items()}
        body_model.reset_params(**body_params)
        model_output = body_model(
            return_verts=True, body_pose=body_params["body_pose"], return_full_pose=True
        )
        vertices = model_output.vertices.detach()

        ## project to image
        calib_path = os.path.join(
            "resource/scan_calibration",
            scene_name,
            "calibration",
            f"{camera_id:03d}.xml",
        )

        cam = CalibratedCamera(calib_path=calib_path)
        projected_vertices = cam(vertices).squeeze().detach().numpy()
        bbox = self.get_bbox(projected_vertices)
        projected_vertices_anatomical = projected_vertices[
            list(AUGMENTED_VERTICES_INDEX_DICT.values())
        ].tolist()

        coco_landmarks = [0] * 3 * 17

        if np.isnan(projected_vertices_anatomical).any():
            return {}, {}, False
        groundtruth_landmarks = {
            name: {"x": point[0], "y": point[1]}
            for name, point in zip(
                AUGMENTED_VERTICES_NAMES, projected_vertices_anatomical
            )
        }

        # check if each landmark is out of frame (visible) or not:
        for name, point in groundtruth_landmarks.items():
            if (
                point["x"] < 0
                or point["y"] < 0
                or point["x"] > 1280
                or point["y"] > 720
            ):
                groundtruth_landmarks[name]["v"] = 0
            else:
                groundtruth_landmarks[name]["v"] = 1

        return groundtruth_landmarks, coco_landmarks, bbox, True

    def generate_dataset(self):
        iteration = 0
        it_file = 0
        nb_files = len(SEQ_NAMES)
        gender_mapping = json.load(open("resource/gender.json", "r"))
        imgext = json.load(open("resource/imgext.json", "r"))

        for seq_name in SEQ_NAMES:
            scene_name, sub_id, _ = seq_name.split("_")
            # extension = imgext[scene_name]
            extension = "jpeg"
            gender = gender_mapping[f"{int(sub_id)}"]

            seq_path = os.path.join(IMAGES_ROOT_PATH, SET, seq_name)
            cams_paths = [
                item
                for item in os.listdir(seq_path)
                if os.path.isdir(os.path.join(seq_path, item))
            ]
            body_model = SMPLX(
                SMPLX_MODEL_DIR,
                gender=gender,
                num_pca_comps=12,
                flat_hand_mean=False,
                create_expression=True,
                create_jaw_pose=True,
            )
            for cam_path in cams_paths:
                camera_id = int(cams_paths[0].split("_")[-1])
                images_paths = glob.glob(
                    os.path.join(seq_path, cam_path) + f"/*.{extension}"
                )
                nb_images = len(images_paths) // self.sample_rate
                for index_frame, image_path in enumerate(images_paths):
                    frame_id = int(image_path.split("/")[-1].split("_")[0])

                    if os.path.exists(
                        os.path.join(
                            "resource/bodies",
                            set,
                            seq_name,
                            f"{frame_id:05d}",
                            f"{sub_id}.pkl",
                        )
                    ):
                        continue
                    if index_frame % self.sample_rate == 0:
                        (
                            groundtruth_landmarks,
                            coco_landmarks,
                            bbox,
                            success,
                        ) = self.get_grountruth_landmarks(
                            body_model,
                            SET,
                            scene_name,
                            seq_name,
                            camera_id,
                            frame_id,
                            sub_id,
                        )

                        self.total_source_images += 1
                        if not success:
                            self.total_error_reconstruction += 1
                            continue

                        annotation_dict = self.generate_annotation_dict()
                        annotation_dict["bbox"] = bbox.tolist()
                        annotation_dict["keypoints"] = groundtruth_landmarks
                        annotation_dict["coco_keypoints"] = coco_landmarks

                        self.data_dict["annotations"].append(annotation_dict)

                        image_dict = {
                            "id": len(self.data_dict["images"]),
                            "width": self.img_width,
                            "height": self.img_height,
                            "frame_number": index_frame,
                            "img_path": image_path,
                        }
                        self.data_dict["images"].append(image_dict)

                        if iteration % 100 == 0:
                            print(
                                f"scene {it_file}/{nb_files}, {index_frame//self.sample_rate}/{nb_images}"
                            )
                        iteration += 1

            it_file += 1

        with open(
            os.path.join(self.output_path, "annotations.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.data_dict, f, ensure_ascii=False, indent=4)

        print("total source images: ", self.total_source_images)
        print("total error reconstruction: ", self.total_error_reconstruction)


if __name__ == "__main__":
    dataset_generator = DatasetGenerator(
        output_path="test_annotations",
        sample_rate=1,
    )
    dataset_generator.generate_dataset()
