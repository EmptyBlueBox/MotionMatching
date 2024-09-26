import numpy as np
import smplx
import rerun as rr
from utils import compute_vertex_normals
import os
import torch

def visualize_motion(motion, config):
    keypoint_list = np.array(config["motion_matching"]["keypoint_list"])
    keypoint_list = np.insert(keypoint_list, 1, 0, axis=1)
    down_sample_rate = config["visualization"]["down_sample_rate"]
    
    # Down sample the motion
    motion["translation"] = motion["translation"][::down_sample_rate]
    motion["orientation"] = motion["orientation"][::down_sample_rate]
    motion["body_pose"] = motion["body_pose"][::down_sample_rate]
    motion["hand_pose"] = motion["hand_pose"][::down_sample_rate]
    
    rr.init("motion_visualization", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)
    
    # Visualize the keypoint
    for i in range(len(keypoint_list)):
        rr.log(f"keypoint_{i}", 
               rr.Points3D(
                   positions=keypoint_list[i],
                   labels=i
               ))

    # Visualize the motion
    smplx_model = smplx.create(model_path=os.environ.get("SMPL_MODEL_PATH"),
                      model_type='smplx',
                      gender='neutral',
                      use_face_contour=False,
                      ext='npz',
                      batch_size=motion["translation"].shape[0])
    
    body_parms = {
        'global_orient': torch.Tensor(motion["orientation"]), # controls the global root orientation
        'body_pose': torch.Tensor(motion["body_pose"]), # controls the body
        'hand_pose': torch.Tensor(motion["hand_pose"]), # controls the finger articulation
        'transl': torch.Tensor(motion["translation"]), # controls the global body position
    }
    smplx_output = smplx_model(**{k: v for k, v in body_parms.items()})
    
    for frame in range(0, motion["translation"].shape[0]):  # Use step_size for downsampling    
        vertices = smplx_output.vertices[frame].detach().cpu().numpy()
        joints = smplx_output.joints[frame].detach().cpu().numpy()
        
        rr.set_time_seconds("stable_time", frame / 120.0 * down_sample_rate)  # Keep original timing
        rr.log("human", 
               rr.Mesh3D(
                   vertex_positions=vertices,
                   triangle_indices=smplx_model.faces,
                   vertex_normals=compute_vertex_normals(vertices, smplx_model.faces)
               ))
        # for i in range(joints.shape[0]):
        #     rr.log(f"joints_{i}", 
        #        rr.Points3D(
        #            positions=joints[i],
        #            labels=i
        #        ))
