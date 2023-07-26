import json
import os
import pickle

import cv2
import torch
import trimesh
from smplx import SMPLX
from utils import CalibratedCamera

## input arguments
SET = "train"
SEQ_NAME = "ParkingLot1_005_pushup2"
SCENE_NAME, SUB_ID, _ = SEQ_NAME.split("_")
FRAME_ID = 150
CAMERA_ID = 0
gender_mapping = json.load(open("resource/gender.json", "r"))
GENDER = gender_mapping[f"{int(SUB_ID)}"]
imgext = json.load(open("resource/imgext.json", "r"))
EXT = imgext[SCENE_NAME]

## SMPLX model
SMPLX_MODEL_DIR = "../models/smplx"
body_model = SMPLX(
    SMPLX_MODEL_DIR,
    gender=GENDER,
    num_pca_comps=12,
    flat_hand_mean=False,
    create_expression=True,
    create_jaw_pose=True,
)

## passing the parameters through SMPL-X
smplx_params_fn = os.path.join(
    "data/bodies", SET, SEQ_NAME, f"{FRAME_ID:05d}", f"{SUB_ID}.pkl"
)
body_params = pickle.load(open(smplx_params_fn, "rb"))
body_params = {k: torch.from_numpy(v) for k, v in body_params.items()}
body_model.reset_params(**body_params)
model_output = body_model(
    return_verts=True, body_pose=body_params["body_pose"], return_full_pose=True
)
mesh = trimesh.Trimesh(
    vertices=model_output.vertices.detach().cpu().squeeze().numpy(),
    faces=body_model.faces,
    process=False,
)
mesh.export("samples/tmp.obj")

## project to image
calib_path = os.path.join(
    "data/scan_calibration", SCENE_NAME, "calibration", f"{CAMERA_ID:03d}.xml"
)
cam = CalibratedCamera(calib_path=calib_path)
j_2D = cam(model_output.joints).squeeze().detach().numpy()
img_fn = os.path.join(
    "data/images",
    SET,
    SEQ_NAME,
    f"cam_{CAMERA_ID:02d}",
    f"{FRAME_ID:05d}_{CAMERA_ID:02d}.{EXT}",
)
img = cv2.imread(img_fn)
for j in j_2D[:25]:
    cv2.circle(img, (int(j[0]), int(j[1])), 6, (255, 0, 255), thickness=-1)
cv2.imwrite("samples/tmp.jpg", img)
