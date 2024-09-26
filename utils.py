import numpy as np
import os
import torch
import smplx
import math
from scipy.spatial.transform import Rotation as R

def compute_vertex_normals(vertices, faces):
    """
    Compute vertex normals using vectorized operations.

    Arguments:
    vertices (np.ndarray): An array of vertex coordinates with shape (N, 3).
    faces (np.ndarray): An array of vertex indices for each face with shape (M, 3).

    Returns:
    np.ndarray: An array of normalized vertex normals with shape (N, 3).
    """
    # Get the vertices of the triangles
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute the normal vectors for each face
    normals = np.cross(v1 - v0, v2 - v0)

    # Compute the lengths of the normal vectors
    norm_lengths = np.linalg.norm(normals, axis=1)

    # Avoid division by zero, set the normal vectors with zero length to a small value
    norm_lengths[norm_lengths == 0] = 1e-10

    # Normalize the normal vectors
    normals /= norm_lengths[:, np.newaxis]

    # Add the normal vectors to the vertices
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], normals)

    # Compute the lengths of the vertex normals
    vertex_norm_lengths = np.linalg.norm(vertex_normals, axis=1)

    # Avoid division by zero, set the normal vectors with zero length to a small value
    vertex_norm_lengths[vertex_norm_lengths == 0] = 1e-10

    # Normalize the vertex normals
    vertex_normals = (vertex_normals.T / vertex_norm_lengths).T
    return vertex_normals


def z_up_to_y_up_translation(translation):
    """
    The function `z_up_to_y_up` performs a transformation on a given translation to
    switch the z-axis and y-axis directions.
    
    Arguments:
    
    * `translation`: [N, 3] ndarray, representing translations
    
    Returns:

    * `translation`: [N, 3] ndarray, representing translations
    """

    translation=translation[:, [0, 2, 1]]*np.array([1, 1, -1])
    return translation


def z_up_to_y_up_rotation(orientation):
    """
    The function `z_up_to_y_up` performs a transformation on a given orientation to
    switch the z-axis and y-axis directions.
    
    Arguments:
    
    * `orientation`: [N, 3] ndarray, representing rotations in rotvec form, with z-up direction
    
    Returns:
    
    * `orientation`: [N, 3] ndarray, representing rotations in rotvec form, with y-up direction
    """
    R_original = R.from_rotvec(orientation)
    R_z_up_to_y_up = R.from_euler('XYZ', [-np.pi/2, 0, 0], degrees=False)
    orientation = R_z_up_to_y_up*R_original
    orientation = orientation.as_rotvec()
    return orientation


def decompose_rotation_with_yaxis(rotation):
    """
    Decompose the rotation into rotation around the y-axis and rotation in the xz-plane.
    
    Arguments:
    
    * `rotation`: [N, 3] ndarray, representing rotations in rotvec form
    
    Returns:
    
    * `root_rot_y`: [N, ] ndarray, representing rotation around the y-axis, in radians
    """
    one_rotation = False
    if rotation.ndim == 1:
        one_rotation = True
        rotation = rotation.reshape(1, -1)

    # Convert rotation vectors to rotation objects
    rot = R.from_rotvec(rotation)
    
    # Get rotation matrices
    matrices = rot.as_matrix()  # Shape: (N, 3, 3)
    
    # Extract the y-axis from each rotation matrix
    yaxis = matrices[:, 1, :]  # Shape: (N, 3)
    
    # Define the global y-axis
    global_y = np.array([0, 1, 0]).reshape(1, 3)  # Shape: (1, 3)
    
    # Compute the dot product between yaxis and global_y for each rotation
    dot_product = np.clip(np.einsum('ij,ij->i', yaxis, global_y), -1.0, 1.0)  # Shape: (N,)
    
    # Calculate the angle between yaxis and global_y
    angles = np.arccos(dot_product)  # Shape: (N,)
    
    # Compute the rotation axis as the cross product between yaxis and global_y
    axes = np.cross(yaxis, global_y)  # Shape: (N, 3)
    
    # Compute the norm of each axis
    axes_norm = np.linalg.norm(axes, axis=1, keepdims=True)  # Shape: (N, 1)
    
    # Normalize the axes, avoiding division by zero
    axes_normalized = np.where(axes_norm > 1e-10, axes / axes_norm, 0.0)  # Shape: (N, 3)
    
    # Create rotation vectors for rotation around the y-axis
    rot_vec = axes_normalized * angles[:, np.newaxis]  # Shape: (N, 3)
    
    # Create inverse rotation objects
    rot_inv = R.from_rotvec(rot_vec).inv()  # Inverse rotations
    
    # Apply inverse rotations to decompose
    Ry = rot_inv * rot  # Rotations around y-axis
    
    # Convert the resulting rotations to rotation vectors
    Ry_rotvec = Ry.as_rotvec()  # Shape: (N, 3)
    
    # Calculate the magnitude of rotation around the y-axis
    # Pay attention to the sign of the y-axis rotation!!!
    Ry_rad = np.linalg.norm(Ry_rotvec, axis=1) * np.sign(Ry_rotvec[:, 1])  # Shape: (N,)
    
    if one_rotation:
        return Ry_rad[0]
    else:
        return Ry_rad


def get_smplx_joint_position(motion_data, select_joint_index):
    """
    Get the feet position of the smplx model
    """
    seq_len=motion_data["translation"].shape[0]
    smplx_model = smplx.create(model_path=os.environ.get("SMPL_MODEL_PATH"),
                      model_type='smplx',
                      gender='neutral',
                      use_face_contour=False,
                      ext='npz',
                      batch_size=seq_len)
    body_parms = {
        'transl': torch.Tensor(motion_data["translation"]), # controls the global body position
        'global_orient': torch.Tensor(motion_data["orientation"]), # controls the global root orientation
        'body_pose': torch.Tensor(motion_data["body_pose"]), # controls the body
        'hand_pose': torch.Tensor(motion_data["hand_pose"]), # controls the finger articulation
    }
    smplx_output = smplx_model(**{k: v for k, v in body_parms.items()})
    
    joint_position=smplx_output.joints[:, select_joint_index].detach().cpu().numpy()
    
    return joint_position


def rotate_xz_vector(angle, vector_2d):
    """
    Rotate the 2D vector in the xz plane
    """
    if vector_2d.ndim == 1:
        R_y = R.from_rotvec(np.array([0, angle, 0]))
        vector_2d = vector_2d.reshape(1, -1)
        vector_3d = np.insert(vector_2d, 1, 0, axis=1)
        vector_3d = R_y.apply(vector_3d)
        return vector_3d[0, [0, 2]]
    else:
        R_y = R.from_rotvec(angle.reshape(-1, 1) * np.array([[0, 1, 0]]))
        vector_3d = np.insert(vector_2d, 1, 0, axis=1)
        vector_3d = R_y.apply(vector_3d)
        return vector_3d[:, [0, 2]]


def halflife2dampling(halflife):
    return 4 * math.log(2) / halflife


def decay_spring_implicit_damping_pos(pos, vel, halflife, dt):
    '''
    A damped spring, used to attenuate position
    '''
    d = halflife2dampling(halflife)/2
    j1 = vel + d * pos
    eydt = math.exp(-d * dt)
    pos = eydt * (pos+j1*dt)
    vel = eydt * (vel - j1 * dt * d)
    return pos, vel


def decay_spring_implicit_damping_rot(rot, avel, halflife, dt):
    '''
    A damped spring, used to attenuate rotation
    '''
    d = halflife2dampling(halflife)/2
    j0 = rot
    j1 = avel + d * j0
    eydt = math.exp(-d * dt)
    a1 = eydt * (j0+j1*dt)
    
    rot_res = R.from_rotvec(a1).as_rotvec()
    avel_res = eydt * (avel - j1 * dt * d)
    return rot_res, avel_res


def concatenate_two_positions(pos1, pos2, frame_time:float = 1/120, half_life:float = 0.2):
    """
    Concatenate two positions with a spring
    
    Arguments:
        * `pos1`: [N, M, 3] ndarray, Indicates the first segment of motion
        * `pos2`: [N, M, 3] ndarray, Indicates the second segment of motion
        * `frame_time`: float, Indicates the time of a frame
        * `half_life`: float, Indicates the half-life of the spring
    
    Returns:
        * `pos`: [N, M, 3] ndarray, Indicates the second segment of motion
    """
    one_joint = False # 是否只对一个关节进行操作
    if pos1.ndim == 2:
        one_joint = True
        pos1 = pos1[:, np.newaxis, :]
        pos2 = pos2[:, np.newaxis, :]

    pos_diff = pos1[-1] - pos2[0]
    v_diff = (pos1[-1] - pos1[-2])/frame_time - (pos2[0] - pos2[1])/frame_time
    
    len2, joint_num, _ = np.shape(pos2)
    for i in range(len2):
        for j in range(joint_num):
            pos_offset, _ = decay_spring_implicit_damping_pos(pos_diff[j], v_diff[j], half_life, i * frame_time)
            pos2[i,j] += pos_offset 

    if one_joint:
        pos2 = pos2[:, 0, :]

    return pos2


def concatenate_two_rotations(rot1, rot2, frame_time:float = 1/120, half_life:float = 0.2):
    """
    Concatenate two rotations with a spring
    
    Arguments:
        * `rot1`: [N, M, 3] ndarray, Indicates the first segment of motion, in the form of rotation axis angle
        * `rot2`: [N, M, 3] ndarray, Indicates the second segment of motion, in the form of rotation axis angle
        * `frame_time`: float, Indicates the time of a frame
        * `half_life`: float, Indicates the half-life of the spring
    
    Returns:
        * `rot`: [N, M, 4] ndarray, Indicates the second segment of motion, in the form of rotation axis angle
    """
    one_joint = False # 是否只对一个关节进行操作
    if rot1.ndim == 2:
        one_joint = True
        rot1 = rot1[:, np.newaxis, :]
        rot2 = rot2[:, np.newaxis, :]

    R12 = R.from_rotvec(rot1[-2])
    R11 = R.from_rotvec(rot1[-1])
    R21 = R.from_rotvec(rot2[0])
    R22 = R.from_rotvec(rot2[1])
    
    rot_diff = (R11*R21.inv()).as_rotvec()
    avel_diff = np.linalg.norm((R11 * R12.inv()).as_rotvec(), axis=1) - np.linalg.norm((R22 * R21.inv()).as_rotvec(), axis=1)
    
    frame_num, joint_num, _ = np.shape(rot2)
    for i in range(frame_num):
        for j in range(joint_num):
            rot_offset, _ = decay_spring_implicit_damping_rot(rot_diff[j], avel_diff[j], half_life, i * frame_time)
            rot2[i,j] = (R.from_rotvec(rot_offset) * R.from_rotvec(rot2[i,j])).as_rotvec()

    if one_joint:
        rot2 = rot2[:, 0, :]

    return rot2 


def interpolate_2d_line_by_distance(points, distance, expected_points=10):
    """
    Perform linear interpolation on a 2D polyline to generate points at a fixed distance,
    ensuring at least a minimum number of points.

    Parameters:
    points (numpy.ndarray): Array of original points with shape (n, 2).
    distance (float): Fixed distance between interpolated points.
    expected_points (int): Expected number of points to ensure.

    Returns:
    numpy.ndarray: Interpolated points.
    """
    # Initialize the list of interpolated points with the first point
    interpolated_points = [points[0]]
    accumulated_distance = 0.0

    # Iterate over each segment in the polyline
    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]
        segment_length = np.linalg.norm(p2 - p1)
        
        # Calculate how many full segments of 'distance' fit in the current segment
        while accumulated_distance + segment_length >= distance:
            # Calculate the position along the segment where the new point should be placed
            t = (distance - accumulated_distance) / segment_length
            new_point = (1 - t) * p1 + t * p2
            interpolated_points.append(new_point)
            if len(interpolated_points) >= expected_points:
                return np.array(interpolated_points)
            # Update p1 to the new point for the next iteration
            p1 = new_point
            # Reset the accumulated distance
            accumulated_distance = 0.0
            # Update segment_length to the remaining part of the segment
            segment_length = np.linalg.norm(p2 - p1)
        
        # Add any remaining distance to the accumulated distance
        accumulated_distance += segment_length

    # Ensure at least the expected number of points
    if len(interpolated_points) < expected_points:
        additional_points = expected_points - len(interpolated_points)
        total_length = sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points)))
        target_distances = np.linspace(0, total_length, expected_points)
        interpolated_points = [points[0]]  # Reset and start with the first point again
        accumulated_distance = 0.0
        
        for i in range(1, len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            segment_length = np.linalg.norm(p2 - p1)
            
            while accumulated_distance + segment_length >= distance:
                t = (distance - accumulated_distance) / segment_length
                new_point = (1 - t) * p1 + t * p2
                interpolated_points.append(new_point)
                p1 = new_point
                accumulated_distance = 0.0
                segment_length = np.linalg.norm(p2 - p1)
            
            accumulated_distance += segment_length

    return np.array(interpolated_points)


if __name__ == "__main__":
    pass