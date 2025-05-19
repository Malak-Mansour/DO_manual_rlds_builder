from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py
import os


class DOManual(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for DOManual HDF5-based teleoperation dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(240, 320, 3),  # 320x240 RGB
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(1,),  # dummy since we dont have wrist images
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='wrist_image: Dummy value for RLDS compliance.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='state: Dummy value for RLDS compliance.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.float32,
                        doc='Robot action consisting of [6x pose euler, 1x tip state, 3x padding or future extension].'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.hdf5'),
            'val': self._generate_examples(path='data/val/episode_*.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        episode_paths = glob.glob(path)

        for episode_path in episode_paths:
            with h5py.File(episode_path, 'r') as f:
                steps = sorted([k for k in f.keys() if k.startswith('step_')], key=lambda x: int(x.split('_')[1]))
                episode = []

                for i, step_key in enumerate(steps):
                    obs_group = f[step_key]['observation']
                    cam_img = obs_group['camera'][()]  # raw image
                    qpose = obs_group['right']['qpose_euler'][()]
                    tip_state = obs_group['right']['tip_state'][()]
                    
                    # Ensure shape compatibility for tfds
                    if cam_img.shape != (240, 320, 3):
                        continue

                    action = np.concatenate([qpose, tip_state, np.zeros(3)], axis=0).astype(np.float32)
                    
                    
                    # language_instruction = "pick up something"
                    # language_embedding = self._embed([language_instruction])[0].numpy()

                    # Extract language instruction from root of HDF5
                    raw_instr = f['language_instruction'][()]
                    if isinstance(raw_instr, bytes):
                        language_instruction = raw_instr.decode('utf-8')
                    else:
                        language_instruction = str(raw_instr)

                    # Embed once for the entire episode
                    language_embedding = self._embed([language_instruction])[0].numpy()



                    episode.append({
                        'observation': {
                            'image': cam_img,
                            'wrist_image': np.zeros((1,), dtype=np.uint8),  # dummy
                            'state': np.zeros((1,), dtype=np.float32)       # dummy
                        },
                        'action': action,
                        'discount': 1.0,
                        'reward': float(i == len(steps) - 1),
                        'is_first': i == 0,
                        'is_last': i == len(steps) - 1,
                        'is_terminal': i == len(steps) - 1,
                        'language_instruction': language_instruction,
                        'language_embedding': language_embedding,
                    })

                yield os.path.basename(episode_path), {
                    'steps': episode,
                    'episode_metadata': {'file_path': episode_path}
                }
