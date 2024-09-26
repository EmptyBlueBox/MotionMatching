from motion_matching import MotionMatching
from visualization import visualize_motion
import yaml

def main():
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    motion_matching = MotionMatching(config)
    motion = motion_matching.run()
    visualize_motion(motion, config)
    
if __name__ == "__main__":
    main()