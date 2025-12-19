import io
import json
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import base64

from openai import OpenAI
import google.generativeai as genai


# -----------------------------
# Prompt utilities
# -----------------------------
def load_prompt(prompt_path: str = "prompt/prompting.json") -> Tuple[str, ...]:
    """
    Load prompt templates from a JSON file and return them as a tuple.
    Using a tuple avoids accidental one-time consumption issues.
    """
    with open(prompt_path, "r") as f:
        prompts_dict = json.load(f)

    keys = [
        "prompt_image0_append_gvl",
        "prompt_image_current_prepend_gvl",
        "prompt_image_prev_prepend_gvl",
        "prompt_image0_append",
        "prompt_image_current_prepend",
        "prompt_image_prev_prepend",
        "prompt_image_prev_append_template",
        "prompt0_template",
        "task_prepend",
        "prompt0_template_gvl",
        "decomposition_examples",
    ]
    return tuple(prompts_dict[k] for k in keys)


def encode_image(image_path: str) -> str:
    """
    Encode the left half of an image file as a base64 string.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        left_half = img.crop((0, 0, width // 2, height))
        buf = io.BytesIO()
        fmt = img.format or "JPEG"
        left_half.save(buf, format=fmt)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


# -----------------------------
# Core GVL
# -----------------------------
def gvl(
    model_name: str,
    task_description_i: str,
    camera_view: str,
    frame_file_list: Sequence[str],
    image_file_num_list: Sequence[int],
    try_count_max: int = 4,
    seed: Optional[int] = None,
    openai_client: Optional[OpenAI] = None,
    gemini_model: Optional[Any] = None,
    prompt_path: str = "prompt/prompting.json",
) -> Tuple[List[int], List[str], List[int]]:
    """
    GVL-style randomized frame querying.

    Design choice:
    - This function predicts progress for frames EXCLUDING the initial frame (index 0).
      The caller can prepend an initial 0 progress and empty description afterwards.

    Returns:
        progress_list: predictions in the SAME ORDER as eval_frame_indices_shuffled
        frame_descriptions_list: descriptions in the SAME ORDER as eval_frame_indices_shuffled
        eval_frame_indices_shuffled: shuffled frame indices (e.g., [7, 2, 10, ...])
    """

    (
        prompt_image0_append_gvl,
        prompt_image_current_prepend_gvl,
        prompt_image_prev_prepend_gvl,
        _prompt_image0_append,
        _prompt_image_current_prepend,
        _prompt_image_prev_prepend,
        _prompt_image_prev_append_template,
        _prompt0_template,
        _task_prepend,
        prompt0_template_gvl,
        _decomposition_examples,
    ) = load_prompt(prompt_path)

    if seed is not None:
        np.random.seed(seed)

    if len(frame_file_list) < 2:
        raise ValueError("frame_file_list must contain at least 2 frames.")

    # Validate clients/models
    if "gpt" in model_name:
        if openai_client is None:
            raise ValueError("openai_client is required when using a GPT model_name.")
    elif "gemini" in model_name:
        if gemini_model is None:
            raise ValueError("gemini_model is required when using a Gemini model_name.")
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Build the base prompt (text-only part)
    prompt0 = prompt0_template_gvl.format(
        task_description=task_description_i,
        camera_view=camera_view.upper(),
    )

    # Use the first frame as the reference frame (image0)
    base64_image0 = encode_image(frame_file_list[0])

    # Evaluate frames excluding the initial frame (index 0)
    num_frames = len(frame_file_list)
    eval_frame_indices = list(range(1, num_frames))

    # Shuffle evaluation frame indices (these are frame indices, not permutation positions)
    eval_frame_indices_shuffled = np.random.permutation(eval_frame_indices).tolist()

    current_progress = 0
    current_frame_description: Optional[str] = None

    progress_list: List[int] = []
    frame_descriptions_list: List[str] = []
    response_text_list: List[str] = []

    # Keep the "conversation" content after the last successful step
    messages_content_success: Optional[list] = None

    for step_idx, frame_idx in enumerate(eval_frame_indices_shuffled):
        base64_image_current = encode_image(frame_file_list[frame_idx])

        response_text: str = ""
        success = False

        for attempt in range(try_count_max):
            try:
                # Build a fresh messages_content for this attempt to avoid message duplication on retries
                if step_idx == 0:
                    # First query: include prompt0, image0, initial progress, and first sampled frame
                    if "gpt" in model_name:
                        messages_content = [
                            {"type": "text", "text": prompt0},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image0}", "detail": "high"},
                            },
                            {
                                "type": "text",
                                "text": prompt_image0_append_gvl.format(current_progress=current_progress),
                            },
                            {
                                "type": "text",
                                "text": prompt_image_current_prepend_gvl.format(current_idx=step_idx + 1),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"},
                            },
                        ]
                    else:
                        messages_content = [
                            prompt0,
                            {"mime_type": "image/jpeg", "data": base64_image0},
                            prompt_image0_append_gvl.format(current_progress=current_progress),
                            prompt_image_current_prepend_gvl.format(current_idx=step_idx + 1),
                            {"mime_type": "image/jpeg", "data": base64_image_current},
                        ]
                else:
                    # Subsequent queries: append previous progress/description and the next sampled frame
                    if messages_content_success is None:
                        raise RuntimeError("Internal error: messages_content_success is None for step_idx > 0.")

                    if "gpt" in model_name:
                        messages_content = list(messages_content_success)  # shallow copy
                        messages_content += [
                            {
                                "type": "text",
                                "text": prompt_image_prev_prepend_gvl.format(
                                    current_progress=current_progress,
                                    current_frame_description=current_frame_description or "",
                                ),
                            },
                            {
                                "type": "text",
                                "text": prompt_image_current_prepend_gvl.format(current_idx=step_idx + 1),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"},
                            },
                        ]
                    else:
                        messages_content = list(messages_content_success)
                        messages_content += [
                            prompt_image_prev_prepend_gvl.format(
                                current_progress=current_progress,
                                current_frame_description=current_frame_description or "",
                            ),
                            prompt_image_current_prepend_gvl.format(current_idx=step_idx + 1),
                            {"mime_type": "image/jpeg", "data": base64_image_current},
                        ]

                # Query the model
                if "gpt" in model_name:
                    resp = openai_client.chat.completions.create(
                        temperature=0.0,
                        model=model_name,
                        messages=[{"role": "user", "content": messages_content}],
                    )
                    response_text = resp.choices[0].message.content or ""
                else:
                    resp = gemini_model.generate_content(
                        messages_content,
                        generation_config=genai.GenerationConfig(
                            max_output_tokens=100,
                            temperature=0,
                            top_k=1,
                        ),
                    )
                    d = resp.to_dict()
                    response_text = d["candidates"][0]["content"]["parts"][0]["text"]

                # Parse the expected output format
                if "Frame description: " not in response_text or "completion percentage: " not in response_text:
                    raise ValueError(f"Unexpected response format: {response_text}")

                current_frame_description = response_text.split("Frame description: ")[1].split("\n")[0]
                progress_str = response_text.split("completion percentage: ")[1].split("%")[0].strip()
                current_progress = int(progress_str)

                # Commit this step
                response_text_list.append(response_text)
                frame_descriptions_list.append(current_frame_description)
                progress_list.append(current_progress)

                # Save the successful messages_content as history for the next step
                messages_content_success = messages_content
                success = True
                break

            except Exception as e:
                print(f"[GVL] Error (step={step_idx}, frame_idx={frame_idx}, attempt={attempt + 1}/{try_count_max}): {e}")
                if response_text:
                    print(f"[GVL] Last response_text: {response_text}")

        if not success:
            # If all retries failed, record a placeholder and continue
            response_text_list.append("")
            frame_descriptions_list.append("")
            progress_list.append(0)

        # Logging (kept similar to your original prints)
        print(step_idx)
        print(frame_idx)
        print(frame_descriptions_list[-1])
        print(progress_list[-1])

    return progress_list, frame_descriptions_list, eval_frame_indices_shuffled


# -----------------------------
# Post-processing
# -----------------------------
def process_gvl_output(
    progress_list: Sequence[int],
    frame_descriptions_list: Sequence[str],
    eval_frame_indices_shuffled: Sequence[int],
    num_frames: Optional[int] = None,
) -> Tuple[List[int], List[str]]:
    """
    Reorder GVL outputs back to chronological order (excluding the initial frame).

    Inputs:
        progress_list / frame_descriptions_list:
            outputs aligned with eval_frame_indices_shuffled order
        eval_frame_indices_shuffled:
            shuffled frame indices that were evaluated (typically 1..num_frames-1)
        num_frames:
            total number of frames in the episode (len(frame_file_list)).
            If None, it will be inferred as max(eval_frame_indices_shuffled)+1.

    Returns:
        final_progress_list:
            per-frame progress in chronological order for frames 1..num_frames-1
        final_frame_descriptions_list:
            per-frame descriptions in chronological order for frames 1..num_frames-1
    """
    if len(progress_list) != len(frame_descriptions_list) or len(progress_list) != len(eval_frame_indices_shuffled):
        raise ValueError("Input lengths must match: progress_list, frame_descriptions_list, eval_frame_indices_shuffled")

    if len(eval_frame_indices_shuffled) == 0:
        return [], []

    if num_frames is None:
        num_frames = int(max(eval_frame_indices_shuffled)) + 1

    # Build mapping from frame_idx -> (progress, description)
    frame_to_pred: dict[int, Tuple[int, str]] = {}
    for p, d, fidx in zip(progress_list, frame_descriptions_list, eval_frame_indices_shuffled):
        frame_to_pred[int(fidx)] = (int(p), str(d))

    # Output for frames 1..num_frames-1
    final_progress_list: List[int] = []
    final_desc_list: List[str] = []

    last_p = 0
    last_d = ""

    for fidx in range(1, num_frames):
        if fidx in frame_to_pred:
            last_p, last_d = frame_to_pred[fidx]
        final_progress_list.append(last_p)
        final_desc_list.append(last_d)

    return final_progress_list, final_desc_list
