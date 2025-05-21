'''
/step_0/  
    action/
        base #3x1
        extra/
            buttons/
        left
        right


        observation/     

            camera/ 
                image #resolution: 320x240   

            right/  

                joint_pose #7x1  
                qpose_euler #6x1  

                qpose_quat #7x1  

                tip_state #1x1 


use rlds_env (the one described in the original repo)

cd /home/malak.mansour/Downloads/ICL/DO_manual_rlds_builder/do_manual
tfds build --overwrite

converted dataset saved at: /home/malak.mansour/tensorflow_datasets
'''

from typing import Iterator, Tuple, Any
import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from do_manual.conversion_utils import MultiThreadedDatasetBuilder

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes from DO_manual dataset paths."""
    for episode_path in paths:
        try:
            with h5py.File(episode_path, "r") as F:
                step_keys = sorted(
                    [k for k in F.keys() if k.startswith("step_")],
                    key=lambda x: int(x.split("_")[1])
                )

                if not step_keys:
                    continue  # Skip empty files

                episode = []
                for i, step_key in enumerate(step_keys):
                    try:
                        obs = F[f"{step_key}/observation"]
                        image = obs["camera/image"][()]
                        joint_pose = obs["right/joint_pose"][()]
                        qpose_euler = obs["right/qpose_euler"][()]
                        qpose_quat = obs["right/qpose_quat"][()]
                        tip_state = obs["right/tip_state"][()]
                        robot_state = np.concatenate([joint_pose, qpose_euler, qpose_quat, tip_state], axis=0).astype(np.float32)


                        '''
                        ACTION extraction: 
                        base and left are all 0s, extra/buttons is empty but doesnt matter, we only want right
                        base (3) + buttons (?) + left (7) + right (7)
                        '''
                        action_group = F[f"{step_key}/action"]
                        action_right = action_group["right"][()]
                        

                        episode.append({
                            'observation': {
                                'image': image,
                                'state': robot_state,
                                'joint_state': joint_pose.astype(np.float32),
                            },
                            'action': action_right.astype(np.float32),
                            'discount': 1.0,
                            'reward': float(i == (len(step_keys) - 1)),
                            'is_first': i == 0,
                            'is_last': i == (len(step_keys) - 1),
                            'is_terminal': i == (len(step_keys) - 1),
                            'language_instruction': "Default instruction",
                        })
                    except Exception as e:
                        print(f"Error processing step {step_key} in {episode_path}: {e}")
                        continue

                if episode:
                    sample = {
                        'steps': episode,
                        'episode_metadata': {
                            'file_path': episode_path
                        }
                    }
                    yield os.path.basename(episode_path), sample
        except Exception as e:
            print(f"Error processing file {episode_path}: {e}")
            continue



class DoManual(MultiThreadedDatasetBuilder):
    """DatasetBuilder for DO_manual dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release for DO_manual dataset.',
    }

    N_WORKERS = 20
    MAX_PATHS_IN_MEMORY = 50
    PARSE_FCN = _generate_examples


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the parse function
        self.PARSE_FCN = _generate_examples

        
    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Camera RGB image (flipped).',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(21,),
                            dtype=np.float32,
                            doc='Concatenated robot state: joint_pose (7), qpose_euler (6), qpose_quat (7), tip_state (1).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Joint pose of the robot arm.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),  # take right only (7)
                        dtype=np.float32,
                        doc='Concatenated action vector: base (3), buttons (?), left arm (7), right arm (7).',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Always 1.0 for demonstration data.',
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward of 1.0 at the final step only.',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if this is the first step of the episode.',
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if this is the last step of the episode.',
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True for terminal steps (last step in demos).',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction for the task (static placeholder).',
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Original file path of the episode.',
                    ),
                }),
            })
        )


            # "train": glob.glob("D:\Malak Doc\Malak Education\MBZUAI\Academic years\Spring 2025\ICL\DO_manual_rlds_builder\do_manual_dataset\DO_manual_dataset\*.hdf5"),
    def _split_paths(self):
        return {
            "train": glob.glob(r"recorded_data_hdf5/*.hdf5"),
        }

