import numpy as np
import os
import hashlib
from scipy.spatial.transform import Rotation as R
from utils import decompose_rotation_with_yaxis, z_up_to_y_up_translation, z_up_to_y_up_rotation, get_smplx_joint_position, rotate_xz_vector

class MotionMatchingDatabase:
    def __init__(self, config):
        self.config = config
        self.AMASS_path = self.config["database"]["AMASS_path"]
        self.AMASS_cache_path = self.config["database"]["AMASS_cache_path"]
        
        # Fixed index for left and right foot
        self.left_foot_index = self.config["database"]["left_foot_index"]
        self.right_foot_index = self.config["database"]["right_foot_index"]
        
        # load motion data
        self.motion_data = {}
        self.load_data()
        
        # calculate key info
        self.motion_data_key = {}
        self.calculate_key()
        
        # loss weight
        self.loss_weight = self.config["database"]["loss_weight"]

    def load_data(self):
        motion_files=os.listdir(self.AMASS_path)
        for motion_file in motion_files:
            motion_file_path = os.path.join(self.AMASS_path, motion_file)
            motion_data_all = np.load(motion_file_path)
            motion_data = {
                "translation": z_up_to_y_up_translation(motion_data_all["trans"]),
                "orientation": z_up_to_y_up_rotation(motion_data_all["poses"][:, :3]),
                "body_pose": motion_data_all["poses"][:, 3:66],
                "hand_pose": motion_data_all["poses"][:, 66:],
                # "beta": motion_data_all["betas"]
            }
            self.motion_data[motion_file] = motion_data
            print(f"Load motion data: {motion_file}")
            
    def calculate_key(self):
        # See if the cache exists, use hash to determine if the cache is valid
        for motion_name, motion_data in self.motion_data.items():
            cache_hash = hashlib.md5(motion_data["translation"].tobytes()).hexdigest()
            print(f"Key cache hash for {motion_name}: {cache_hash}")
            
            # Check if the cache exists
            if os.path.exists(os.path.join(self.AMASS_cache_path, f"{motion_name.split('.')[0]}.npz")):
                key_cache=np.load(os.path.join(self.AMASS_cache_path, f"{motion_name.split('.')[0]}.npz"))
                key_cache_hash = key_cache["hash"]
                if key_cache_hash == cache_hash:
                    print(f"Load key cache for {motion_name}")  
                    self.motion_data_key[motion_name] = key_cache
                    continue

            # If the cache does not exist, calculate the key info
            print(f"No key cache for {motion_name}, calculate key info")
            
            # Calculate root xz-plane translation, rotation, velocity, angular velocity
            root_xz_translation = motion_data["translation"][:, [0, 2]]
            root_y_rotation = decompose_rotation_with_yaxis(motion_data["orientation"])
            root_xz_velocity = np.gradient(root_xz_translation, axis=0)
            root_xz_velocity = rotate_xz_vector(-root_y_rotation, root_xz_velocity) # 将xz平面的速度旋转到 y 朝前
            root_y_angular_velocity = np.gradient(root_y_rotation, axis=0)
            
            # Calculate left and right foot relative position, y rotation is front
            foot_relative_position = get_smplx_joint_position(motion_data, [self.left_foot_index, self.right_foot_index]) - motion_data["translation"][:,None,:]
            R_y = R.from_rotvec(root_y_rotation.reshape(-1, 1) * np.array([[0, 1, 0]]))
            front_left_foot_relative_position = R_y.inv().apply(foot_relative_position[:, 0])
            front_right_foot_relative_position = R_y.inv().apply(foot_relative_position[:, 1])
            foot_relative_position = np.concatenate([front_left_foot_relative_position[:,None,:], front_right_foot_relative_position[:,None,:]], axis=1)
            
            self.motion_data_key[motion_name] = {
                "hash": cache_hash,
                "root_xz_translation": root_xz_translation, # 保存根节点的位置是因为需要有将来的期望路径约束
                "root_xz_velocity": root_xz_velocity,
                "root_y_rotation": root_y_rotation, # 保存根节点的旋转, 虽然没有将来的角度约束, 但是需要旋转路径
                "root_y_angular_velocity": root_y_angular_velocity,
                "foot_relative_position": foot_relative_position,
            }
            
            # Save the key info to the cache
            np.savez(os.path.join(self.AMASS_cache_path, f"{motion_name.split('.')[0]}.npz"), **self.motion_data_key[motion_name])
            
            print(f"Save key cache for {motion_name}")
    
    def print_database_info(self):
        for motion_name, motion_data in self.motion_data.items():
            print(f"Motion data index: {motion_name}, raw data length: {motion_data['translation'].shape[0]}, key data length: {self.motion_data_key[motion_name]['root_xz_translation'].shape[0]}")
            print(f'Raw data key: {list(self.motion_data_key[motion_name].keys())}')
            print(f'Key data: {list(self.motion_data_key[motion_name].keys())}')
    
    def search(self, condition, search_interval=20):
        '''
        Args:
            - condition: a dict containing the condition for searching
                - cur_root_velocity: the current root velocity, one value
                - cur_root_y_angular_velocity: the current root y angular velocity, one value
                - cur_foot_relative_position: the current foot relative position, shape (2, 3)
                - future_root_xz_position: the future root xz position (relative to the current frame, also rotated so that the current root y rotation is 0), shape (2, )
            - search_interval: the search interval

        Returns:
            - motion: a dict containing the following several frames motion data, **start from 0 pos and 0 orientation**
                - translation: the translation of the motion, shape (search_interval, 3)
                - orientation: the orientation of the motion, shape (search_interval, 3)
                - body_pose: the body pose of the motion, shape (search_interval, 63)
                - hand_pose: the hand pose of the motion, shape (search_interval, 99)
        '''
        min_loss = float('inf')
        best_motion_name = None
        best_frame_index = None
        for motion_name, motion_data_key in self.motion_data_key.items():
            # Calculate current loss
            front_root_xz_velocity = motion_data_key["root_xz_velocity"]
            loss_cur_root_xz_velocity = np.linalg.norm(front_root_xz_velocity - condition["cur_root_xz_velocity"], axis=1)[:-search_interval]
            loss_cur_root_y_angular_velocity = np.abs(motion_data_key["root_y_angular_velocity"] - condition["cur_root_y_angular_velocity"])[:-search_interval]
            
            # Calculate current foot relative position loss
            # print(f"motion_data_key['foot_relative_position']: {motion_data_key['foot_relative_position'].shape}")
            # print(f"condition['cur_foot_relative_position']: {condition['cur_foot_relative_position'].shape}")
            loss_cur_foot_relative_position = np.sum(np.linalg.norm(motion_data_key["foot_relative_position"] - condition["cur_foot_relative_position"], axis=2), axis=1)[:-search_interval]
            
            # Calculate future root relative xz position
            future_root_xz_position = motion_data_key["root_xz_translation"][search_interval:] - motion_data_key["root_xz_translation"][:-search_interval]
            future_root_xz_position_in_current_root_y_rotation = rotate_xz_vector(-motion_data_key["root_y_rotation"][:-search_interval], future_root_xz_position)
            
            # Calculate future loss
            loss_future_root_xz_position = np.linalg.norm(future_root_xz_position_in_current_root_y_rotation - condition["future_root_xz_position"], axis=1)
            
            # Calculate total loss
            loss = loss_cur_root_xz_velocity * self.loss_weight["cur_root_xz_velocity"] + loss_cur_root_y_angular_velocity * self.loss_weight["cur_root_y_angular_velocity"] + loss_cur_foot_relative_position * self.loss_weight["cur_foot_relative_position"] + loss_future_root_xz_position * self.loss_weight["future_root_xz_position"]
            
            # Find the best motion
            tmp_best_loss = np.min(loss)
            if tmp_best_loss < min_loss:
                min_loss = tmp_best_loss
                best_motion_name = motion_name
                best_frame_index = np.argmin(loss)
                
                print(f"Best motion: {best_motion_name}, frame index: {best_frame_index}")
                print(f"cur_root_xz_velocity_loss: {loss_cur_root_xz_velocity[best_frame_index]*self.loss_weight['cur_root_xz_velocity']:.6f},\n\
cur_root_y_angular_velocity_loss: {loss_cur_root_y_angular_velocity[best_frame_index]*self.loss_weight['cur_root_y_angular_velocity']:.6f},\n\
cur_foot_relative_position_loss: {loss_cur_foot_relative_position[best_frame_index]*self.loss_weight['cur_foot_relative_position']:.6f},\n\
future_root_xz_position_loss: {loss_future_root_xz_position[best_frame_index]*self.loss_weight['future_root_xz_position']:.6f}")
        
        R_y = R.from_rotvec(self.motion_data_key[best_motion_name]["root_y_rotation"][best_frame_index].reshape(-1, 1) * np.array([[0, 1, 0]])) # Reverse the rotation so that the root node of the current frame faces forward.
        front_translation = R_y.inv().apply(self.motion_data[best_motion_name]["translation"][best_frame_index:best_frame_index+search_interval] - self.motion_data[best_motion_name]["translation"][best_frame_index])
        front_orientation = (R_y.inv()*R.from_rotvec(self.motion_data[best_motion_name]["orientation"][best_frame_index:best_frame_index+search_interval])).as_rotvec()
        # ## test
        # R_0=R.from_rotvec(self.motion_data[best_motion_name]["orientation"][best_frame_index])
        # print(f'R_0: {R_0.as_quat()}')
        # RRR=R.from_rotvec(decompose_rotation_with_yaxis(self.motion_data[best_motion_name]["orientation"][best_frame_index])*np.array([0,1,0]))
        # print(f'Ry: {RRR.as_rotvec()}')
        # print(f'decompose no Ry: {decompose_rotation_with_yaxis((RRR.inv()*R.from_rotvec(self.motion_data[best_motion_name]["orientation"][best_frame_index])).as_rotvec())}')
        # ## test
        motion = {
            "translation": front_translation,
            "orientation": front_orientation,
            "body_pose": self.motion_data[best_motion_name]["body_pose"][best_frame_index:best_frame_index+search_interval],
            "hand_pose": self.motion_data[best_motion_name]["hand_pose"][best_frame_index:best_frame_index+search_interval],
            
            # For update current state, don't need to calculate again
            "root_xz_velocity": self.motion_data_key[best_motion_name]["root_xz_velocity"][best_frame_index:best_frame_index+search_interval],
            "root_y_angular_velocity": self.motion_data_key[best_motion_name]["root_y_angular_velocity"][best_frame_index:best_frame_index+search_interval],
            "foot_relative_position": self.motion_data_key[best_motion_name]["foot_relative_position"][best_frame_index:best_frame_index+search_interval],
        }
        return motion
                
                