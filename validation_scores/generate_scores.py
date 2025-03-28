#%%
import os
from cellmap_flow.image_data_interface import ImageDataInterface
from pathlib import Path
import zarr
datasets = {
    "jrc_c-elegans-bw-1": {
        "mito": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/mito_filled",
        "ld": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/ld_filled",
        "lyso": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/lyso_filled",
        "perox": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/perox_filled",
        "nuc": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/nuc_filled",
        "yolk": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/yolk_filled"
    },
    "jrc_c-elegans-comma-1": {
        "mito": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1.zarr/mito_filled",
        "ld": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1.zarr/ld_filled",
        "lyso": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1.zarr/lyso_filled",
        "perox": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1.zarr/perox_filled",
        "nuc": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1.zarr/nuc_filled",
        "yolk": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1.zarr/yolk_filled"
    },
    "jrc_c-elegans-op50-1": {
        "mito": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/mito_filled",
        "ld": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/ld_filled",
        "lyso": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/lyso_filled",
        "perox": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/perox_filled",
        "nuc": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/nuc_filled",
        "yolk": "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/yolk_filled"
    }
}

gt_crops = { "jrc_c-elegans-op50-1": "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/labels/groundtruth/",
         "jrc_c-elegans-bw-1": "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/labels/groundtruth/",
  "jrc_c-elegans-comma-1":"/nrs/cellmap/zubovy/crop_splits/c-elegans/8nm/combined/jrc_c-elegans-comma-1/groundtruth.zarr/"
#   "jrc_c-elegans-comma-1":"/nrs/cellmap/zubovy/crop_splits/c-elegans/8nm/combined/jrc_c-elegans-comma-1/groundtruth.zarr/crop497/"
}
#%%
def get_scores(pred, label):
    pred = (pred.flatten()>0).astype(int)
    label = (label.flatten()>0).astype(int)
    tp = ((pred == 1) & (label == 1)).sum()
    tn = ((pred == 0) & (label == 0)).sum()
    fp = ((pred == 1) & (label == 0)).sum()
    fn = ((pred == 0) & (label == 1)).sum()
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "iou": tp / (tp + fp + fn),
        "dice": 2 * tp / (2 * tp + fp + fn),
    }

#%%
scores = {}
for dataset, organelles in datasets.items():
    scores[dataset] = {}
    crops = [f for f in os.listdir(gt_crops[dataset]) if os.path.isdir(os.path.join(gt_crops[dataset], f))]
    for crop in crops:
        scores[dataset][crop] = {}
        crop_path = os.path.join(gt_crops[dataset], crop)
        for org, org_path in organelles.items():
            # print(f"Processing {dataset} {org}")
            result_ds = ImageDataInterface(
                os.path.join(org_path,"s0"), normalize=False
            )
            for i in range(3):
                label_crop = ImageDataInterface(
                    os.path.join(crop_path,org,f"s{i}")
                )
                if label_crop.voxel_size == result_ds.voxel_size:
                    label_data = zarr.open(os.path.join(crop_path,org,f"s{i}") )[:]
                    break
            # print(result_ds.voxel_size)
            # print(label_crop.voxel_size)
            # print(label_crop.roi)
            result_data = result_ds.to_ndarray_ts(label_crop.roi)
            # label_data = label_crop.to_ndarray_ts()
            # print(result_data.shape)
            # print(label_data.shape)
            sc = get_scores(result_data, label_data)
            print(f"score {org}: {sc}")
            scores[dataset][crop][org] = sc

        # break
#%%
scores
# %%
f1_scores = {}
for dataset, crops in scores.items():
    f1_scores[dataset] = {}
    for crop, organelles in crops.items():
        f1_scores[dataset][crop] = {}
        for org, org_scores in organelles.items():
            f1_scores[dataset][crop][org] = float(org_scores["f1"])
# %%
f1_scores
# %%
for dataset, organelles in f1_scores.items():
    print(f"{dataset}:")
    for org, org_scores in organelles.items():
        print(f"{org}: {org_scores:.3f}")
# %%
from funlib.evaluate import rand_voi
# %%

instances_scores = {}
for dataset, organelles in datasets.items():
    instances_scores[dataset] = {}
    crops = [f for f in os.listdir(gt_crops[dataset]) if os.path.isdir(os.path.join(gt_crops[dataset], f))]
    for crop in crops:
        instances_scores[dataset][crop] = {}
        crop_path = os.path.join(gt_crops[dataset], crop)
        for org, org_path in organelles.items():
            # print(f"Processing {dataset} {org}")
            result_ds = ImageDataInterface(
                os.path.join(org_path,"s0"), normalize=False
            )
            for i in range(3):
                label_crop = ImageDataInterface(
                    os.path.join(crop_path,org,f"s{i}")
                )
                if label_crop.voxel_size == result_ds.voxel_size:
                    label_data = zarr.open(os.path.join(crop_path,org,f"s{i}") )[:].astype(np.uint64)
                    break
            # print(result_ds.voxel_size)
            # print(label_crop.voxel_size)
            # print(label_crop.roi)
            result_data = result_ds.to_ndarray_ts(label_crop.roi).astype(np.uint64)
            # label_data = label_crop.to_ndarray_ts()
            # print(result_data.shape)
            # print(label_data.shape)
            sc = rand_voi(result_data, label_data)
            print(f"score {org}: {sc}")
            instances_scores[dataset][crop][org] = sc

# %%
instances_scores
# %%
def voi_score(voi_split, voi_merge):
    return (voi_split + voi_merge) / 2
voi_scores = {}
for dataset, organelles in instances_scores.items():
    voi_scores[dataset] = {}
    for org, org_scores in organelles.items():
        voi_scores[dataset][org] = voi_score(org_scores["voi_merge"], org_scores["voi_split"])
# %%
voi_scores
# %%
import matplotlib.pyplot as plt
def visualize(dataset,organelle):
    result_ds = ImageDataInterface(
        os.path.join(datasets[dataset][organelle],"s0"), normalize=False
    )
    for i in range(3):
        label_crop = ImageDataInterface(
            os.path.join(crops[dataset],organelle,f"s{i}")
        )
        if label_crop.voxel_size == result_ds.voxel_size:
            label_data = zarr.open(os.path.join(crops[dataset],organelle,f"s{i}") )[:].astype(np.uint64)
            break
    # print(result_ds.voxel_size)
    # print(label_crop.voxel_size)
    # print(label_crop.roi)
    result_data = result_ds.to_ndarray_ts(label_crop.roi).astype(np.uint64)
    # label_data = label_crop.to_ndarray_ts()
    print(result_data.shape)
    print(label_data.shape)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(result_data[100]>0, cmap="gray")
    ax[0].set_title("Prediction")
    ax[1].imshow(label_data[100]>0, cmap="gray")
    ax[1].set_title("Label")
    ax[2].imshow(result_data[100] == label_data[100], cmap="gray")
    ax[2].set_title("Comparison")
    plt.show()
# %%
visualize("jrc_c-elegans-op50-1","mito")
# %%
visualize("jrc_c-elegans-bw-1","mito")

# %%
import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as relabel
from pykdtree.kdtree import KDTree as cKDTree
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)

HAUSDORFF_DISTANCE_MAX = np.inf
CAST_TO_NONE = [np.nan, np.inf, -np.inf]

MAX_INSTANCE_THREADS = int(os.getenv("MAX_INSTANCE_THREADS", 2))
MAX_SEMANTIC_THREADS = int(os.getenv("MAX_SEMANTIC_THREADS", 22))
PER_INSTANCE_THREADS = int(os.getenv("PER_INSTANCE_THREADS", 8))
# submitted_# of instances / ground_truth_# of instances
INSTANCE_RATIO_CUTOFF = float(os.getenv("INSTANCE_RATIO_CUTOFF", 100))
PRECOMPUTE_LIMIT = int(os.getenv("PRECOMPUTE_LIMIT", 1e8))
DEBUG = os.getenv("DEBUG", "False") != "False"


def optimized_hausdorff_distances(
    truth_label,
    matched_pred_label,
    voxel_size,
    hausdorff_distance_max,
    method="standard",
):
    # Get unique truth IDs, excluding the background (0)
    truth_ids = np.unique(truth_label)
    truth_ids = truth_ids[truth_ids != 0]  # Exclude background
    if len(truth_ids) == 0:
        return []

    def get_distance(i):
        # Skip if both masks are empty
        truth_mask = truth_label == truth_ids[i]
        pred_mask = matched_pred_label == truth_ids[i]
        if not np.any(truth_mask) and not np.any(pred_mask):
            return 0

        # Compute Hausdorff distance for the current pair
        h_dist = compute_hausdorff_distance(
            truth_mask,
            pred_mask,
            voxel_size,
            hausdorff_distance_max,
            method,
        )
        return i, h_dist

    # Initialize list for distances
    hausdorff_distances = np.empty(len(truth_ids))
    if DEBUG:
        # Use tqdm for progress tracking
        bar = tqdm(
            range(len(truth_ids)),
            desc="Computing Hausdorff distances",
            leave=True,
            dynamic_ncols=True,
            total=len(truth_ids),
        )
        # Compute the cost matrix
        for i in bar:
            i, h_dist = get_distance(i)
            hausdorff_distances[i] = h_dist
    else:
        with ThreadPoolExecutor(max_workers=PER_INSTANCE_THREADS) as executor:
            # with ProcessPoolExecutor(max_workers=PER_INSTANCE_THREADS) as executor:
            for i, h_dist in tqdm(
                executor.map(get_distance, range(len(truth_ids))),
                desc="Computing Hausdorff distances",
                total=len(truth_ids),
                dynamic_ncols=True,
            ):
                hausdorff_distances[i] = h_dist

    return hausdorff_distances


def compute_hausdorff_distance(image0, image1, voxel_size, max_distance, method):
    """
    Compute the Hausdorff distance between two binary masks, optimized for pre-vectorized inputs.
    """
    # Extract nonzero points
    a_points = np.argwhere(image0)
    b_points = np.argwhere(image1)

    # Handle empty sets
    if len(a_points) == 0 and len(b_points) == 0:
        return 0
    elif len(a_points) == 0 or len(b_points) == 0:
        return np.inf

    # Scale points by voxel size
    a_points = a_points * np.array(voxel_size)
    b_points = b_points * np.array(voxel_size)

    # Build KD-trees once
    a_tree = cKDTree(a_points)
    b_tree = cKDTree(b_points)

    # Query distances
    # fwd = a_tree.query(b_points, k=1, distance_upper_bound=max_distance)[0]
    # bwd = b_tree.query(a_points, k=1, distance_upper_bound=max_distance)[0]
    fwd = a_tree.query(b_points, k=1)[0]
    bwd = b_tree.query(a_points, k=1)[0]

    # Replace "inf" with `max_distance` for numerical stability
    # fwd[fwd == np.inf] = max_distance
    # bwd[bwd == np.inf] = max_distance
    fwd[fwd > max_distance] = max_distance
    bwd[bwd > max_distance] = max_distance

    if method == "standard":
        return max(fwd.max(), bwd.max())
    elif method == "modified":
        return max(fwd.mean(), bwd.mean())
    
class spoof_precomputed:
    def __init__(self, array, ids):
        self.array = array
        self.ids = ids
        self.index = -1

    def __getitem__(self, ids):
        if isinstance(ids, int):
            return np.array(self.array == self.ids[ids], dtype=bool)
        return np.array([self.array == self.ids[i] for i in ids], dtype=bool)

    def __len__(self):
        return len(self.ids)
    
def score_instance(
    pred_label,
    truth_label,
    voxel_size = (16,16,16),
    hausdorff_distance_max=HAUSDORFF_DISTANCE_MAX,
) -> dict[str, float]:
    """
    Score a single instance label volume against the ground truth instance label volume.

    Args:
        pred_label (np.ndarray): The predicted instance label volume.
        truth_label (np.ndarray): The ground truth instance label volume.
        voxel_size (tuple): The size of a voxel in each dimension.
        hausdorff_distance_max (float): The maximum distance to consider for the Hausdorff distance.

    Returns:
        dict: A dictionary of scores for the instance label volume.

    Example usage:
        scores = score_instance(pred_label, truth_label)
    """
    logging.info("Scoring instance segmentation...")
    # Relabel the predicted instance labels to be consistent with the ground truth instance labels
    logging.info("Relabeling predicted instance labels...")
    pred_label = relabel(pred_label, connectivity=len(pred_label.shape))

    # Get unique IDs, excluding background (assumed to be 0)
    truth_ids = np.unique(truth_label)
    truth_ids = truth_ids[truth_ids != 0]

    pred_ids = np.unique(pred_label)
    pred_ids = pred_ids[pred_ids != 0]

    # Skip if the submission has way too many instances
    if len(truth_ids) > 0 and len(pred_ids) / len(truth_ids) > INSTANCE_RATIO_CUTOFF:
        logging.warning(
            f"WARNING: Skipping {len(pred_ids)} instances in submission, {len(truth_ids)} in ground truth, because there are too many instances in the submission."
        )
        return {
            "accuracy": 0,
            "hausdorff_distance": np.inf,
            "normalized_hausdorff_distance": 0,
            "combined_score": 0,
        }

    # Initialize the cost matrix
    logging.info(
        f"Initializing cost matrix of {len(truth_ids)} x {len(pred_ids)} (true x pred)..."
    )
    cost_matrix = np.zeros((len(truth_ids), len(pred_ids)))

    # Flatten the labels for vectorized computation
    truth_flat = truth_label.flatten()
    pred_flat = pred_label.flatten()

    # Precompute binary masks for all `truth_ids`
    if len(truth_flat) * len(truth_ids) > PRECOMPUTE_LIMIT:
        truth_binary_masks = spoof_precomputed(truth_flat, truth_ids)
    else:
        logging.info("Precomputing binary masks for all `truth_ids`...")
        truth_binary_masks = np.array(
            [(truth_flat == tid) for tid in truth_ids], dtype=bool
        )

    def get_cost(j):
        # Find all `truth_ids` that overlap with this prediction mask
        pred_mask = pred_flat == pred_ids[j]
        relevant_truth_ids = np.unique(truth_flat[pred_mask])
        relevant_truth_ids = relevant_truth_ids[relevant_truth_ids != 0]
        relevant_truth_indices = np.where(np.isin(truth_ids, relevant_truth_ids))[0]
        relevant_truth_masks = truth_binary_masks[relevant_truth_indices]

        if relevant_truth_indices.size == 0:
            return [], j, []

        tp = relevant_truth_masks[:, pred_mask].sum(1)
        fn = (relevant_truth_masks[:, pred_mask == 0]).sum(1)
        fp = (relevant_truth_masks[:, pred_mask] == 0).sum(1)

        # Compute Jaccard scores
        jaccard_scores = tp / (tp + fp + fn)

        # Fill in the cost matrix for this `j` (prediction)
        return relevant_truth_indices, j, jaccard_scores

    matched_pred_label = np.zeros_like(pred_label)

    if len(pred_ids) > 0:
        # Compute the cost matrix
        if DEBUG:
            # Use tqdm for progress tracking
            bar = tqdm(
                range(pred_ids),
                desc="Computing cost matrix",
                leave=True,
                dynamic_ncols=True,
                total=len(pred_ids),
            )
            # Compute the cost matrix
            for j in bar:
                relevant_truth_indices, j, jaccard_scores = get_cost(j)
                cost_matrix[relevant_truth_indices, j] = jaccard_scores
        else:
            with ThreadPoolExecutor(max_workers=PER_INSTANCE_THREADS) as executor:
                # with ProcessPoolExecutor(max_workers=PER_INSTANCE_THREADS) as executor:
                for relevant_truth_indices, j, jaccard_scores in tqdm(
                    executor.map(get_cost, range(len(pred_ids))),
                    desc="Computing cost matrix in parallel",
                    dynamic_ncols=True,
                    total=len(pred_ids),
                    leave=True,
                ):
                    cost_matrix[relevant_truth_indices, j] = jaccard_scores

        # Match the predicted instances to the ground truth instances
        logging.info("Calculating linear sum assignment...")
        row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=True)

        # Contruct the volume for the matched instances
        for i, j in tqdm(
            zip(col_inds, row_inds),
            desc="Relabeling matched instances",
            dynamic_ncols=True,
        ):
            if pred_ids[i] == 0 or truth_ids[j] == 0:
                # Don't score the background
                continue
            pred_mask = pred_label == pred_ids[i]
            matched_pred_label[pred_mask] = truth_ids[j]

        hausdorff_distances = optimized_hausdorff_distances(
            truth_label, matched_pred_label, voxel_size, hausdorff_distance_max
        )
    else:
        # No predictions to match
        hausdorff_distances = []

    # Compute the scores
    logging.info("Computing accuracy score...")
    accuracy = accuracy_score(truth_label.flatten(), matched_pred_label.flatten())
    hausdorff_dist = np.mean(hausdorff_distances) if len(hausdorff_distances) > 0 else 0
    normalized_hausdorff_dist = 1.01 ** (
        -hausdorff_dist / np.linalg.norm(voxel_size)
    )  # normalize Hausdorff distance to [0, 1] using the maximum distance represented by a voxel. 32 is arbitrarily chosen to have a reasonable range
    combined_score = (accuracy * normalized_hausdorff_dist) ** 0.5
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Hausdorff Distance: {hausdorff_dist:.4f}")
    logging.info(f"Normalized Hausdorff Distance: {normalized_hausdorff_dist:.4f}")
    logging.info(f"Combined Score: {combined_score:.4f}")
    return {
        "accuracy": accuracy,
        "hausdorff_distance": hausdorff_dist,
        "normalized_hausdorff_distance": normalized_hausdorff_dist,
        "combined_score": combined_score,
    }
# %%
# from funlib.evaluate import detection_scores

instances_scores = {}
for dataset, organelles in datasets.items():
    instances_scores[dataset] = {}
    crops = [f for f in os.listdir(gt_crops[dataset]) if os.path.isdir(os.path.join(gt_crops[dataset], f))]
    for crop in crops:
        instances_scores[dataset][crop] = {}
        crop_path = os.path.join(gt_crops[dataset], crop)
        for org, org_path in organelles.items():
            # print(f"Processing {dataset} {org}")
            result_ds = ImageDataInterface(
                os.path.join(org_path,"s0"), normalize=False
            )
            for i in range(3):
                label_crop = ImageDataInterface(
                    os.path.join(crop_path,org,f"s{i}")
                )
                if label_crop.voxel_size == result_ds.voxel_size:
                    label_data = zarr.open(os.path.join(crop_path,org,f"s{i}") )[:].astype(np.uint64)
                    break
            # print(result_ds.voxel_size)
            # print(label_crop.voxel_size)
            # print(label_crop.roi)
            result_data = result_ds.to_ndarray_ts(label_crop.roi).astype(np.uint64)
            # label_data = label_crop.to_ndarray_ts()
            # print(result_data.shape)
            # print(label_data.shape)
            sc = score_instance(result_data, label_data)
            print(f"score {org}: {sc}")
            instances_scores[dataset][crop][org] = sc
instances_scores
# %%
# save det_scores to a json file
import json
with open("/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/validation_scores/all_instances_scores.json", "w") as f:
    json.dump(instances_scores, f)
# %%
scores
# %%
import json
# Convert numpy types to native Python types
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

with open("/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/validation_scores/all_crops_semantic_scores.json", "w") as f:
    json.dump(scores, f, default=convert_numpy)
# %%
accuracy_scores = {}
for dataset, organelles in instances_scores.items():
    accuracy_scores[dataset] = {}
    for crop, organelles in organelles.items():
        accuracy_scores[dataset][crop] = {}
        for org, org_scores in organelles.items():
            accuracy_scores[dataset][crop][org] = float(org_scores["accuracy"])
# %%
accuracy_scores
# %%
for dataset, organelles in accuracy_scores.items():
    print(f"{dataset}:")
    for org, org_scores in organelles.items():
        print(f"{org}: {org_scores:.3f}")
# %%
