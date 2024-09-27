import numpy as np
import rerun as rr
import smplx
import torch
import os
from utils import compute_vertex_normals
from scipy.spatial.transform import Rotation as R

viz_len = 500  # You can adjust this to your desired number of frames
scale = 5 # select one frame every 5 frames
    
def main():
    # Load AMASS data from NPZ file
    bdata = np.load('./AMASS/0005_Walking001_stageii.npz')
    print(bdata.files)
    
    poses = bdata['poses']
    print(f'poses shape: {poses.shape}')
    
    poses = poses[:100].reshape(100, -1, 3)
    
    real_len = len(bdata['trans'])

    comp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    
    # limit the length of the motion to the length of the visualization
    real_len = min(viz_len*scale, len(bdata['trans']))
    
    # convert z-up to y-up
    transl = bdata['trans'][:real_len:scale]
    transl=transl[:, [0, 2, 1]]*np.array([1, 1, -1])
    
    orientation = bdata['poses'][:real_len:scale, :3]
    R_original = R.from_rotvec(orientation)
    R_z_up_to_y_up = R.from_euler('XYZ', [-np.pi/2, 0, 0], degrees=False)
    orientation = R_z_up_to_y_up*R_original
    orientation = orientation.as_rotvec()
    
    body_parms = {
        'global_orient': torch.Tensor(orientation).to(comp_device), # controls the global root orientation
        'body_pose': torch.Tensor(bdata['poses'][:real_len:scale, 3:66]).to(comp_device), # controls the body
        'hand_pose': torch.Tensor(bdata['poses'][:real_len:scale, 66:]).to(comp_device), # controls the finger articulation
        'transl': torch.Tensor(transl).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=viz_len, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    }

    print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    print(f'viz frame length = {viz_len}')
    print(f'real frame length = {real_len}')
    
    # Visualization
    rr.init("amass_visualization", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)

    smplx_model = smplx.create(model_path=os.environ.get("SMPL_MODEL_PATH"),
                      model_type='smplx',
                      gender='neutral',
                      use_face_contour=False,
                      num_betas=num_betas,
                      num_dmpls=num_dmpls,
                      ext='npz',
                      batch_size=viz_len)
    faces = smplx_model.faces

    smplx_output = smplx_model(**{k: v.to(comp_device) for k, v in body_parms.items()})
    
    for frame in range(0, viz_len):  # Use step_size for downsampling    
        vertices = smplx_output.vertices[frame].detach().cpu().numpy()
        joints = smplx_output.joints[frame].detach().cpu().numpy()
        
        rr.set_time_seconds("stable_time", frame / 120.0 * scale)  # Keep original timing
        rr.log("human", 
               rr.Mesh3D(
                   vertex_positions=vertices,
                   triangle_indices=faces,
                   vertex_normals=compute_vertex_normals(vertices, faces)
               ))
        for i in range(joints.shape[0]):
            rr.log(f"joints_{i}", 
               rr.Points3D(
                   positions=joints[i],
                   labels=i
               ))


if __name__ == "__main__":
    main()