'''
/episode_0/
    o └── actions: (N,7): concatenated ee_states+gripper_states
    o └──teleop_actions: (N,7): relative action
    o └── obs/
        o ├── agentview_rgb: shape (N, H, W, 3)
        o ├── eye_in_hand_rgb: shape (N, H, W, 3)
        o ├── gripper_states: shape (N,1)
        o └── ee_states: shape (N, 6), robot absolute pose
        o └── joint_positions: shape (N, 6)


        
conda activate rlds_env #the one described in the original repo
cd /home/malak.mansour/Downloads/ICL/DO_manual_rlds_builder/do_manual

tfds build --overwrite
tfds build --data_dir=/l/users/malak.mansour/Datasets/do_manual/rlds/ --overwrite


converted dataset saved at: /l/users/malak.mansour/Datasets/do_manual/rlds/

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
    """Yields episodes from DO_manual dataset paths using new format."""
    for episode_path in paths:
        try:
            with h5py.File(episode_path, "r") as F:
                for ep_key in F.keys():
                    if not ep_key.startswith("episode_"):
                        continue

                    episode_grp = F[ep_key]
                    actions = episode_grp["actions"][:]
                    teleop_actions = episode_grp.get("teleop_actions", None)
                    obs = episode_grp["obs"]

                    num_steps = actions.shape[0]
                    episode = []

                    for i in range(num_steps):
                        try:
                            step_data = {
                                'observation': {
                                    'agentview_rgb': obs["agentview_rgb"][i],
                                    'eye_in_hand_rgb': obs["eye_in_hand_rgb"][i],
                                    'ee_states': obs["ee_states"][i].astype(np.float32),
                                    'gripper_states': obs["gripper_states"][i].astype(np.float32),
                                    'joint_positions': obs["joint_positions"][i].astype(np.float32),
                                },
                                'action': actions[i].astype(np.float32),
                                'discount': 1.0,
                                'reward': float(i == (num_steps - 1)),
                                'is_first': i == 0,
                                'is_last': i == (num_steps - 1),
                                'is_terminal': i == (num_steps - 1),
                                'language_instruction': os.path.splitext(os.path.basename(episode_path))[0],
                            }


                            if teleop_actions is not None:
                                step_data['teleop_actions'] = teleop_actions[i].astype(np.float32)

                            episode.append(step_data)

                        except Exception as e:
                            print(f"Error processing step {i} in {ep_key} of {episode_path}: {e}")
                            continue

                    if episode:
                        sample = {
                            'steps': episode,
                            'episode_metadata': {
                                'file_path': episode_path
                            }
                        }
                        yield f"{os.path.basename(episode_path)}_{ep_key}", sample

        except Exception as e:
            print(f"Error processing file {episode_path}: {e}")
            continue


class DoManual(MultiThreadedDatasetBuilder):
    """DatasetBuilder for DO_manual dataset using new HDF5 format."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Updated to support new episode-based format for DO_manual dataset.',
    }

    N_WORKERS = 20
    MAX_PATHS_IN_MEMORY = 50
    PARSE_FCN = _generate_examples
    BUILDER_CONFIGS_DIR = "/l/users/malak.mansour/Datasets/do_manual/rlds/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'agentview_rgb': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Third-person camera view RGB image.',
                        ),
                        'eye_in_hand_rgb': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist-mounted camera RGB image.',
                        ),
                        'ee_states': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='6D end-effector absolute pose.',
                        ),
                        'gripper_states': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='Gripper state (open/close).',
                        ),
                        'joint_positions': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Joint positions of the robot.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Absolute action vector: ee_states + gripper.',
                    ),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                    'teleop_actions': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Optional relative action vector.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            })
        )

    def _split_paths(self):
        return {
            "train": glob.glob(r"/l/users/malak.mansour/Datasets/do_manual/hdf5/*.h5"),
        }