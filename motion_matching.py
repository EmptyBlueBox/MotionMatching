import numpy as np
from database import MotionMatchingDatabase
import yaml
from scipy.spatial.transform import Rotation as R
from visualization import visualize_motion
from utils import rotate_xz_vector, decompose_rotation_with_yaxis, concatenate_two_positions, concatenate_two_rotations, interpolate_2d_line_by_distance

class MotionMatching:
    def __init__(self, config):
        # init motion matching
        self.init_seq_name = config["motion_matching"]["init_seq_name"]
        self.init_seq_idx = config["motion_matching"]["init_seq_idx"]
        
        # target
        self.keypoint_list = np.array(config["motion_matching"]["keypoint_list"])
        
        # motion matching hyper-parameters
        self.search_interval = config["motion_matching"]["search_interval"]
        self.consider_trajectory_interval = config["motion_matching"]["consider_trajectory_interval"]
        self.consider_trajectory_range = config["motion_matching"]["consider_trajectory_range"]
        self.frame_time = 1/config["database"]["fps"]
        self.touch_threshold = config["motion_matching"]["touch_threshold"]
        self.half_life = config["motion_matching"]["half_life"]
        self.max_iter = config["motion_matching"]["max_iter"]
        
        # Initialize database
        self.database = MotionMatchingDatabase(config)
        
        # Initialize current state
        self.cur_xyz_translation = np.array([self.keypoint_list[0][0], self.database.motion_data[self.init_seq_name]["translation"][self.init_seq_idx][1], self.keypoint_list[0][1]])
        self.cur_root_xz_translation = self.keypoint_list[0]
        self.cur_root_y_rotation = np.arctan2(
            self.keypoint_list[1][0] - self.keypoint_list[0][0],
            self.keypoint_list[1][1] - self.keypoint_list[0][1]
        )
        self.cur_root_xz_velocity = np.linalg.norm(self.database.motion_data_key[self.init_seq_name]["root_xz_velocity"][self.init_seq_idx]) * (self.keypoint_list[1] - self.keypoint_list[0])/np.linalg.norm(self.keypoint_list[1] - self.keypoint_list[0])
        self.cur_root_y_angular_velocity = 0
        self.cur_foot_relative_position = self.database.motion_data_key[self.init_seq_name]["foot_relative_position"][self.init_seq_idx]
        
        # Initialize target keypoint index
        self.cur_target_keypoint_idx = 1
        
        # Initialize answer
        self.ans = {
            "translation": [],
            "orientation": [],
            "body_pose": [],
            "hand_pose": [],
        }
        
    def controller(self):
        '''
        Return:
            - future_root_xz_translation: the next 10, 20, ..., 60 frames keypoints
        '''
        # trajectory = np.concatenate([self.cur_root_xz_translation[None, :], self.keypoint_list[self.cur_target_keypoint_idx:]], axis=0)
        # distance = np.linalg.norm(self.cur_root_xz_velocity) * self.frame_time
        
        # expected_trajectory = interpolate_2d_line_by_distance(trajectory, distance, expected_points=self.consider_trajectory_range)
        # expected_trajectory = expected_trajectory[1:self.consider_trajectory_range:self.consider_trajectory_interval]

        expected_trajectory = np.stack([self.keypoint_list[self.cur_target_keypoint_idx]]*(self.consider_trajectory_range//self.consider_trajectory_interval), axis=0)
        return expected_trajectory
    
    def smooth_motion(self):
        for i in range(len(self.ans["translation"]) - 1):
            second_motion_length = self.ans["translation"][i + 1].shape[0]
            self.ans["translation"][i + 1] = concatenate_two_positions(self.ans["translation"][i], self.ans["translation"][i + 1])
            self.ans["orientation"][i + 1] = concatenate_two_rotations(self.ans["orientation"][i], self.ans["orientation"][i + 1])
            self.ans["body_pose"][i + 1] = concatenate_two_rotations(self.ans["body_pose"][i].reshape(second_motion_length, -1, 3), self.ans["body_pose"][i + 1].reshape(second_motion_length, -1, 3), frame_time=self.frame_time, half_life=self.half_life).reshape(second_motion_length, -1)
            # self.ans["hand_pose"][i + 1] = concatenate_two_rotations(self.ans["hand_pose"][i].reshape(second_motion_length, -1, 3), self.ans["hand_pose"][i + 1].reshape(second_motion_length, -1, 3), frame_time=self.frame_time, half_life=self.half_life).reshape(second_motion_length, -1)
            
    def run(self):
        '''
        Run motion matching, until the agent arrives at the final target keypoint
        
        Return:
            - self.ans: the motion matching result
                - translation: the translation of the motion
                - orientation: the orientation of the motion
                - body_pose: the body pose of the motion
                - hand_pose: the hand pose of the motion
        '''
        MAX_ITER = self.max_iter
        while np.linalg.norm(self.cur_root_xz_translation - self.keypoint_list[-1]) > self.touch_threshold:
            MAX_ITER -= 1
            if MAX_ITER <= 0:
                print("Max iteration reached, stop motion matching")
                break
            
            condition = {
                "cur_root_xz_velocity": self.cur_root_xz_velocity, # the velocity to the current root y rotation
                "cur_root_y_angular_velocity": self.cur_root_y_angular_velocity,
                "cur_foot_relative_position": self.cur_foot_relative_position,
                "future_root_xz_position": rotate_xz_vector(-self.cur_root_y_rotation, self.controller() - self.cur_root_xz_translation), # negative y rotation
            }
            
            motion = self.database.search(condition)
            
            # Rotate forward so that the y-axis of the current frame changes from facing forward to the correct direction
            R_y = R.from_rotvec(self.cur_root_y_rotation*np.array([[0, 1, 0]]))
            correct_translation = self.cur_xyz_translation + R_y.apply(motion["translation"])
            correct_orientation = (R_y*R.from_rotvec(motion["orientation"])).as_rotvec()
            
            self.ans["translation"].append(correct_translation)
            self.ans["orientation"].append(correct_orientation)
            self.ans["body_pose"].append(motion["body_pose"])
            self.ans["hand_pose"].append(motion["hand_pose"])
            
            self.cur_xyz_translation = correct_translation[-1]
            self.cur_root_xz_translation = correct_translation[-1, [0, 2]]
            self.cur_root_y_rotation = decompose_rotation_with_yaxis(correct_orientation[-1][None, :])[0]
            self.cur_root_xz_velocity = motion["root_xz_velocity"][-1]
            self.cur_root_y_angular_velocity = motion["root_y_angular_velocity"][-1]
            self.cur_foot_relative_position = motion["foot_relative_position"][-1]
            
            print(f"Target keypoint: {self.keypoint_list[self.cur_target_keypoint_idx]}")

            # If arrive at the next target keypoint, move to the next target keypoint
            if np.linalg.norm(self.cur_root_xz_translation - self.keypoint_list[self.cur_target_keypoint_idx]) < self.touch_threshold:
                self.cur_target_keypoint_idx += 1
            
            print(f"cur_root_xz_translation: {self.cur_root_xz_translation}\n")
        
        
        # Smooth the motion before concatenate all motion clips
        self.smooth_motion()

        # Concatenate all motion clips
        self.ans["translation"] = np.concatenate(self.ans["translation"], axis=0)
        self.ans["orientation"] = np.concatenate(self.ans["orientation"], axis=0)
        self.ans["body_pose"] = np.concatenate(self.ans["body_pose"], axis=0)
        self.ans["hand_pose"] = np.concatenate(self.ans["hand_pose"], axis=0)
        print(f'total motion length: {self.ans["translation"].shape[0]}')
        
        return self.ans
    
def main():
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    motion_matching = MotionMatching(config)
    
if __name__ == "__main__":
    main()