'''
To verify that the data is converted correctly, please run the data visualization script from the base directory:

python3 visualize_dataset.py do_manual
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension. 
Note, if you are running on a headless server you can modify WANDB_ENTITY at the top of visualize_dataset.py and 
add your own WandB entity -- then the script will log all visualizations to WandB.
'''

import argparse
import tqdm
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np

# Use non-GUI backend for matplotlib (headless servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb


WANDB_ENTITY = None
WANDB_PROJECT = 'vis_rlds'

# Argument: dataset name
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='name of the dataset to visualize')
args = parser.parse_args()

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY,
               project=WANDB_PROJECT)
else:
    render_wandb = False


# create TF dataset
dataset_name = args.dataset_name
print(f"📊 Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, split='train')
ds = ds.shuffle(100)

# Create output directory for plots
output_dir = "visualization_outputs"
os.makedirs(output_dir, exist_ok=True)

# visualize episodes
for i, episode in enumerate(ds.take(5)):
    images = []
    for step in episode['steps']:
        images.append(step['observation']['image'].numpy())
    image_strip = np.concatenate(images[::4], axis=1)
    caption = step['language_instruction'].numpy().decode() + ' (temp. downsampled 4x)'

    if render_wandb:
        wandb.log({f'image_{i}': wandb.Image(image_strip, caption=caption)})
    else:
        plt.figure()
        plt.imshow(image_strip)
        plt.title(caption)
        plt.axis('off')
        plt.savefig(f"{output_dir}/episode_image_{i}.png")
        plt.close()

# Visualize action and state statistics
actions, states = [], []
for episode in tqdm.tqdm(ds.take(500), desc="🔍 Collecting stats"):
    for step in episode['steps']:
        actions.append(step['action'].numpy())
        states.append(step['observation']['state'].numpy())
actions = np.array(actions)
states = np.array(states)
action_mean = actions.mean(0)
state_mean = states.mean(0)

print("\n📈 Action means per dimension:", action_mean)
print("📈 State means per dimension:", state_mean)

# Plot histograms
def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem+1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(f"{tag} dim {elem}\nMean: {vector_mean[elem]:.3f}")

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})
    else:
        fig.savefig(f"{output_dir}/{tag}.png")
        plt.close(fig)

vis_stats(actions, action_mean, 'action_stats')
vis_stats(states, state_mean, 'state_stats')

if not render_wandb:
    print(f"\n✅ Visualization images saved to: `{output_dir}/`")