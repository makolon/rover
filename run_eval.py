import os
import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eval import (
    eval_frame_level_progress_prediction,
    eval_frame_level_reasoning,
    eval_video_qa,
    LLMCaller,
)
from rover_model import rover, process_rover_output, build_openai_client, build_gemini_model


# -----------------------------
# Configuration
# -----------------------------
dataset_dir = "./data"
api_key = os.environ.get("API_KEY", None)  # Prefer environment variable over hard-coding
method = "rover"
model_name = "gemini-robotics-er-1.5-preview"
task = "PnPCounterToCab"
camera_view = "external"
downsample_to = 30
max_episodes = 1  # Your original code effectively evaluated 2 episodes
level_filter_substring = "lev2"  # Empty string means no filtering

if api_key is None:
    raise ValueError("API key is missing. Set API_KEY env var or assign `api_key` directly.")

openai_client = None
gemini_model = None
if "gpt" in model_name:
    openai_client = build_openai_client(api_key)
elif "gemini" in model_name:
    gemini_model = build_gemini_model(model_name, api_key)
else:
    raise ValueError(f"Unsupported model_name: {model_name}")

# Create LLMCaller for evaluation functions
llm_caller = LLMCaller(
    model_name=model_name,
    openai_client=openai_client,
    gemini_model=gemini_model,
)


# -----------------------------
# Utilities
# -----------------------------
def parse_int_from_frame_path(path: str) -> int:
    """Extract frame index from a path like .../frame_123.jpg."""
    name = os.path.basename(path)
    stem = name.split(".")[0]
    idx_str = stem.split("_")[-1]
    return int(idx_str)


def make_downsampled_frame_list(frame_dir: str, downsample_to_: int) -> Tuple[List[str], List[int]]:
    """
    Build a downsampled list of frame paths and their corresponding integer indices.

    Returns:
        frame_file_list: list of paths, length ~= downsample_to_
        frame_indices: list of frame numbers extracted from filenames, same length as frame_file_list
    """
    frame_file_list_all = glob.glob(os.path.join(frame_dir, "frame_*.jpg"))
    if len(frame_file_list_all) == 0:
        raise FileNotFoundError(f"No frames found under: {frame_dir}")

    frame_nums_all = sorted(parse_int_from_frame_path(p) for p in frame_file_list_all)
    max_frame_num = frame_nums_all[-1]

    # Use linspace in the numeric frame-id space, then ensure uniqueness and sorting
    chosen_nums = np.linspace(0, max_frame_num, downsample_to_).astype(int).tolist()
    chosen_nums = sorted(set(chosen_nums))

    frame_file_list = [os.path.join(frame_dir, f"frame_{n}.jpg") for n in chosen_nums]
    # Filter out missing files (some datasets have sparse numbering)
    frame_file_list = [p for p in frame_file_list if os.path.exists(p)]
    frame_indices = [parse_int_from_frame_path(p) for p in frame_file_list]

    if len(frame_file_list) < 2:
        raise ValueError(f"Not enough frames after downsampling in: {frame_dir}")

    return frame_file_list, frame_indices


def load_gt_progress(gt_progress_file: str) -> List[float]:
    """Load per-frame ground-truth progress from task_progress.txt."""
    if not os.path.exists(gt_progress_file):
        raise FileNotFoundError(f"GT progress file missing: {gt_progress_file}")

    vals: List[float] = []
    with open(gt_progress_file, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(float(s))
    return vals


def select_gt_progress(gt_all: List[float], frame_indices: List[int]) -> List[int]:
    """
    Select GT progress values corresponding to the chosen frame indices.

    This assumes gt_all is indexed by the original frame number.
    If that assumption does not hold in your dataset, adjust here.
    """
    max_needed = max(frame_indices)
    if max_needed >= len(gt_all):
        raise IndexError(
            f"GT progress length ({len(gt_all)}) is smaller than max frame index ({max_needed}). "
            "Your GT indexing convention likely differs; fix select_gt_progress()."
        )

    gt_selected = [gt_all[i] for i in frame_indices]
    gt_selected = [int(round(x)) for x in gt_selected]
    return gt_selected


def get_perturb_info(perturb_info_file: str) -> Tuple[Any, ...]:
    """
    Parse perturbation info file.

    Note: This function is kept compatible with your original return signature,
    but the implementation is simplified and more defensive.
    """
    defaults = (
        None, None, None, None, None, None,  # idx_start..idx_contact_expert
        "",                                  # task_description
        [], [], [], [],                      # dist_list, step_label_list, gripper_target_dist_list, env_dist_list
        None, None,                          # obj_is_touching_gripper_list, obj_is_only_touching_gripper_list
    )

    if not os.path.exists(perturb_info_file):
        return defaults

    idx_start = idx_final = None
    idx_start_contact = idx_contact = None
    idx_start_contact_expert = idx_contact_expert = None
    task_description = ""
    dist_list: List[float] = []
    step_label_list: List[str] = []
    gripper_target_dist_list: List[float] = []
    env_dist_list: List[float] = []
    obj_is_touching_gripper_list: Optional[List[str]] = None
    obj_is_only_touching_gripper_list: Optional[List[str]] = None

    with open(perturb_info_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("idx_start:"):
                idx_start = int(line.split(":")[-1].strip())
            elif line.startswith("idx_final"):
                idx_final = int(line.split(":")[-1].strip())
            elif line.startswith("idx_start_contact:"):
                val = line.split(":")[-1].strip()
                idx_start_contact = None if val == "None" else int(val)
            elif line.startswith("idx_contact:"):
                val = line.split(":")[-1].strip()
                idx_contact = None if val == "None" else int(val)
            elif line.startswith("idx_start_contact_expert:"):
                idx_start_contact_expert = int(line.split(":")[-1].strip())
            elif line.startswith("idx_contact_expert:"):
                idx_contact_expert = int(line.split(":")[-1].strip())
            elif line.startswith("language"):
                task_description = line.split(":", 1)[1].strip().lower()

            elif line.startswith("dist_list:"):
                raw = line.split(": ", 1)[1].strip()
                raw = raw[1:-1]  # strip [ ]
                dist_list = [float(x) for x in raw.split(", ") if x]

            elif "step_label" in line:
                raw = line.split(": ", 1)[1].strip()
                raw = raw[1:-1]
                # step labels are quoted strings in your original code
                step_label_list = [x.strip()[1:-1] for x in raw.split(", ") if x.strip()]

            elif line.startswith("gripper_target_dist_list"):
                raw = line.split(": ", 1)[1].strip()
                raw = raw[1:-1]
                gripper_target_dist_list = [float(x) for x in raw.split(", ") if x]

            elif line.startswith("env_dist_list"):
                raw = line.split(": ", 1)[1].strip()
                raw = raw[1:-1]
                env_dist_list = [float(x) for x in raw.split(", ") if x]

            elif line.startswith("obj_is_touching_gripper_list"):
                raw = line.split(": ", 1)[1].strip()
                if "None" in raw:
                    obj_is_touching_gripper_list = None
                else:
                    raw = raw[1:-1]
                    obj_is_touching_gripper_list = [x for x in raw.split(", ") if x]

            elif line.startswith("obj_is_only_touching_gripper_list"):
                raw = line.split(": ", 1)[1].strip()
                if "None" in raw:
                    obj_is_only_touching_gripper_list = None
                else:
                    raw = raw[1:-1]
                    obj_is_only_touching_gripper_list = [x for x in raw.split(", ") if x]

    return (
        idx_start, idx_final, idx_start_contact, idx_contact,
        idx_start_contact_expert, idx_contact_expert,
        task_description,
        dist_list, step_label_list, gripper_target_dist_list, env_dist_list,
        obj_is_touching_gripper_list, obj_is_only_touching_gripper_list,
    )


# -----------------------------
# Main evaluation loop
# -----------------------------
episode_dir_list = glob.glob(os.path.join(dataset_dir, task, "*"))
episode_dir_list = [p for p in episode_dir_list if os.path.isdir(p)]
episode_dir_list = sorted(episode_dir_list)

# Apply the explicit level filter (your original code did this inside the loop and could crash)
if level_filter_substring:
    episode_dir_list = [p for p in episode_dir_list if level_filter_substring in os.path.basename(p)]

if len(episode_dir_list) == 0:
    raise FileNotFoundError(f"No episode directories found under: {os.path.join(dataset_dir, task)}")

episode_dir_list = episode_dir_list[:max_episodes]

results: List[Dict[str, Any]] = []

# Optional: store extra rover outputs
rover_artifacts: Dict[str, Any] = {
    "final_idx_list": [],
    "subtask_list_list": [],
    "subtask_progress_list_list": [],
    "subtask_frame_descriptions_list_list": [],
    "gt_progress_list_list": [],
    "final_progress_list_list": [],
    "frame_descriptions_list_list": [],
}

for episode_dir in episode_dir_list:
    print("\n" + "*" * 80)
    print(f"Episode: {episode_dir}")

    frame_dir = os.path.join(episode_dir, "frames")
    frame_file_list, frame_indices = make_downsampled_frame_list(frame_dir, downsample_to)

    gt_progress_file = os.path.join(episode_dir, "task_progress.txt")
    gt_all = load_gt_progress(gt_progress_file)
    gt_progress_list = select_gt_progress(gt_all, frame_indices)

    perturb_info_file = os.path.join(episode_dir, "pertub_info.txt")  # Keeping dataset spelling
    (
        idx_start_i, idx_final_i, idx_start_contact_i, idx_contact_i,
        idx_start_contact_expert_i, idx_contact_expert_i,
        task_description_i,
        dist_list_i, step_label_list_i, gripper_target_dist_list_i, env_dist_list_i,
        obj_is_touching_gripper_list_i, obj_is_only_touching_gripper_list_i,
    ) = get_perturb_info(perturb_info_file)

    # Fallback if language line is missing
    if not task_description_i:
        task_description_i = task.lower()

    # Run the selected method
    if method == "rover":
        # NOTE: This assumes rover_model.rover accepts injected clients (openai_client / gemini_model).
        final_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, _ = rover(
            model_name=model_name,
            task_description_i=task_description_i,
            camera_view=camera_view,
            frame_file_list=frame_file_list,
            openai_client=openai_client,
            gemini_model=gemini_model,
        )
        final_progress_list, frame_descriptions_list = process_rover_output(
            subtask_list=subtask_list,
            subtask_progress_list=subtask_progress_list,
            subtask_frame_descriptions_list=subtask_frame_descriptions_list,
            frame_file_list=frame_file_list,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # The rover post-processing returns per-frame predictions excluding the initial frame.
    # Add the initial frame back to match GT length convention used in your eval.
    final_progress_list = [0] + list(final_progress_list)
    frame_descriptions_list = [""] + list(frame_descriptions_list)

    # Basic sanity checks
    if len(final_progress_list) != len(frame_file_list):
        raise ValueError(
            f"Pred progress length ({len(final_progress_list)}) != num frames ({len(frame_file_list)})."
        )
    if len(gt_progress_list) != len(frame_file_list):
        raise ValueError(
            f"GT progress length ({len(gt_progress_list)}) != num frames ({len(frame_file_list)})."
        )

    # Eval task 1
    corr, dist = eval_frame_level_progress_prediction(gt_progress_list, final_progress_list)

    # Eval task 2 - skip if not enough data
    try:
        error_rate, success_rate, inconclusive_rate = eval_frame_level_reasoning(
            frame_descriptions_list,
            llm_caller,
            task_description=task_description_i,
            level_i=os.path.basename(episode_dir),
            task=task,
            idx_start_contact=idx_start_contact_i,
            idx_contact=idx_contact_i,
            step_label_list_ds=step_label_list_i,
            frame_idx_list_ds=frame_indices,
            gripper_target_dist_list=gripper_target_dist_list_i,
            env_dist_list=env_dist_list_i,
            obj_is_touching_gripper_list=obj_is_touching_gripper_list_i,
            obj_is_only_touching_gripper_list=obj_is_only_touching_gripper_list_i,
        )
    except Exception as e:
        print(f"Warning: eval_frame_level_reasoning failed: {e}")
        error_rate, success_rate, inconclusive_rate = 0.0, 0.0, 0.0

    # Eval task 3 - skip if not enough data
    try:
        qa_accuracy, qa_precision, qa_recall, qa_frame_diff = eval_video_qa(
            frame_descriptions_list,
            llm_caller,
            task_description=task_description_i,
            task=task,
            level_i=os.path.basename(episode_dir),
            frame_idx_list_ds=frame_indices,
            idx_final_i=idx_final_i,
            idx_start_contact_i=idx_start_contact_i,
            idx_contact_i=idx_contact_i,
            idx_start_contact_expert_i=idx_start_contact_expert_i,
            idx_contact_expert_i=idx_contact_expert_i,
        )
    except Exception as e:
        print(f"Warning: eval_video_qa failed: {e}")
        qa_accuracy, qa_precision, qa_recall, qa_frame_diff = 0.0, 0.0, 0.0, 0.0

    results.append(
        {
            "episode_dir": os.path.basename(episode_dir),
            "task": task,
            "method": method,
            "model_name": model_name,
            "corr": corr,
            "dist": dist,
            "reasoning_error_rate": error_rate,
            "reasoning_success_rate": success_rate,
            "reasoning_inconclusive_rate": inconclusive_rate,
            "qa_accuracy": qa_accuracy,
            "qa_precision": qa_precision,
            "qa_recall": qa_recall,
            "qa_frame_diff": qa_frame_diff,
        }
    )

    # Save artifacts if rover is used
    if method == "rover":
        rover_artifacts["final_idx_list"].append(final_idx)
        rover_artifacts["subtask_list_list"].append(subtask_list)
        rover_artifacts["subtask_progress_list_list"].append(subtask_progress_list)
        rover_artifacts["subtask_frame_descriptions_list_list"].append(subtask_frame_descriptions_list)

    rover_artifacts["gt_progress_list_list"].append(gt_progress_list)
    rover_artifacts["final_progress_list_list"].append(final_progress_list)
    rover_artifacts["frame_descriptions_list_list"].append(frame_descriptions_list)


# -----------------------------
# Results dataframe
# -----------------------------
res_df = pd.DataFrame(results)
print(res_df)
