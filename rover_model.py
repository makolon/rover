import os
import io
import json
from typing import Any, Optional, Tuple, List

from PIL import Image
import base64

from openai import OpenAI
import google.generativeai as genai


# -----------------------------
# Model/client initialization
# -----------------------------
def build_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Build an OpenAI client instance.
    If api_key is None, it will read OPENAI_API_KEY from environment variables.
    """
    return OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))


def build_gemini_model(model_name: str, api_key: Optional[str] = None) -> Any:
    """
    Build a Gemini GenerativeModel instance.
    If api_key is None, it will read GOOGLE_API_KEY from environment variables.
    """
    genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
    return genai.GenerativeModel(model_name)


# -----------------------------
# Prompt utilities
# -----------------------------
def load_prompt() -> Tuple[str, ...]:
    """
    Load prompt templates from prompt/prompting.json and return them as a tuple.
    Using a tuple (instead of a generator) prevents accidental one-time consumption.
    """
    with open("prompt/prompting.json", "r") as json_file:
        prompts_dict = json.load(json_file)

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


def encode_image(image_path: str, max_size: int = 512, quality: int = 85) -> str:
    """
    Encode the left half of an image as base64 string.
    Resizes image to reduce token usage for API calls.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) in pixels
        quality: JPEG quality (1-95, lower = smaller file)
    """
    with Image.open(image_path) as img:
        width, height = img.size
        left_half = img.crop((0, 0, width // 2, height))
        
        # Resize if image is too large
        left_width, left_height = left_half.size
        if left_width > max_size or left_height > max_size:
            if left_width > left_height:
                new_width = max_size
                new_height = int(left_height * (max_size / left_width))
            else:
                new_height = max_size
                new_width = int(left_width * (max_size / left_height))
            left_half = left_half.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        # Force JPEG format with compression
        left_half.convert('RGB').save(buffered, format='JPEG', quality=quality, optimize=True)
        buffered.seek(0)
        return base64.b64encode(buffered.read()).decode("utf-8")


def flatten(nested):
    """
    Flatten a nested list structure.
    """
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


# -----------------------------
# Core rover
# -----------------------------
def rover(
    model_name: str,
    task_description_i: str,
    camera_view: str,
    frame_file_list: List[str],
    current_start_idx: int = 0,
    subtask_completion_threshold: int = 100,
    include_last_subtask_frame: bool = False,
    openai_client: Optional[OpenAI] = None,
    gemini_model: Optional[Any] = None,
):
    """
    Run the ROVER logic over a frame list.

    IMPORTANT:
    - For OpenAI models (e.g., 'gpt-...'), pass openai_client.
    - For Gemini models (e.g., 'gemini-...'), pass gemini_model.
    """

    (
        _prompt_image0_append_gvl,
        _prompt_image_current_prepend_gvl,
        _prompt_image_prev_prepend_gvl,
        prompt_image0_append,
        prompt_image_current_prepend,
        prompt_image_prev_prepend,
        prompt_image_prev_append_template,
        prompt0_template,
        task_prepend,
        _prompt0_template_gvl,
        decomposition_examples,
    ) = load_prompt()

    # Defensive checks to prevent index errors
    if not frame_file_list:
        return current_start_idx, [], [], [], None

    if current_start_idx < 0 or current_start_idx >= len(frame_file_list):
        raise IndexError(f"current_start_idx out of range: {current_start_idx}")

    current_idx = current_start_idx + 1
    if current_idx >= len(frame_file_list):
        # No "current" frame exists to compare; return safely.
        base64_image0 = encode_image(frame_file_list[current_start_idx])
        return current_idx, [], [], [], base64_image0

    prompt0 = prompt0_template.format(
        task_description=task_description_i,
        decomposition_examples=decomposition_examples,
        camera_view=camera_view.upper(),
    )

    base64_image0 = encode_image(frame_file_list[current_start_idx])
    base64_image_current = encode_image(frame_file_list[current_idx])

    base64_image_prev = None
    prev_progress = None

    progress_list: List[int] = []
    frame_description_list: List[str] = []
    response_text_list: List[str] = []

    subtask_list = []
    subtask_progress_list = []
    subtask_frame_descriptions_list = []
    subtask_idx = -1

    # Build the initial message payload
    if "gpt" in model_name:
        if openai_client is None:
            raise ValueError("openai_client is required when using a GPT model_name.")
        messages_content = [
            {"type": "text", "text": prompt0},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image0}", "detail": "high"}},
            {"type": "text", "text": prompt_image0_append},
            {"type": "text", "text": prompt_image_current_prepend},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"}},
        ]
    elif "gemini" in model_name:
        if gemini_model is None:
            raise ValueError("gemini_model is required when using a Gemini model_name.")
        messages_content = [
            prompt0,
            {"mime_type": "image/jpeg", "data": base64_image0},
            prompt_image0_append,
            prompt_image_current_prepend,
            {"mime_type": "image/jpeg", "data": base64_image_current},
        ]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    while True:
        # Stop condition: out of frames
        if current_idx >= len(frame_file_list):
            return current_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, base64_image_current

        # One VLM query per loop iteration
        if "gpt" in model_name:
            response = openai_client.chat.completions.create(
                temperature=0.0,
                model=model_name,
                messages=[{"role": "user", "content": messages_content}],
            )
            response_text = response.choices[0].message.content or ""
        else:
            response = gemini_model.generate_content(
                messages_content,
                generation_config=genai.GenerationConfig(max_output_tokens=2048, temperature=0, top_k=1),
            )
            d = response.to_dict()
            
            # Safely extract response text with error handling
            if "candidates" in d and len(d["candidates"]) > 0:
                candidate = d["candidates"][0]
                finish_reason = candidate.get("finish_reason", "UNKNOWN")
                finish_reason_names = {
                    0: "FINISH_REASON_UNSPECIFIED",
                    1: "STOP",
                    2: "MAX_TOKENS",
                    3: "SAFETY",
                    4: "RECITATION",
                    5: "OTHER"
                }
                finish_reason_name = finish_reason_names.get(finish_reason, f"UNKNOWN({finish_reason})")
                
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        response_text = parts[0]["text"]
                    else:
                        print(f"Warning: Empty parts in response at frame {current_idx}")
                        print(f"Finish reason: {finish_reason_name}")
                        print(f"Candidate: {candidate}")
                        response_text = ""
                else:
                    print(f"Warning: No content/parts in candidate at frame {current_idx}")
                    print(f"Finish reason: {finish_reason_name}")
                    print(f"Candidate: {candidate}")
                    response_text = ""
            else:
                print(f"Warning: No candidates in response at frame {current_idx}")
                print(f"Response dict: {d}")
                response_text = ""

        response_text_list.append(response_text)
        print("Current Index:", current_idx)
        
        # Handle empty responses to avoid infinite loops
        if not response_text or response_text.strip() == "":
            print(f"Warning: Empty response at frame {current_idx}, skipping to next frame")
            current_idx += 1
            if current_idx < len(frame_file_list):
                base64_image_prev = base64_image_current
                base64_image_current = encode_image(frame_file_list[current_idx])
                # Rebuild messages for next frame
                if "gpt" in model_name:
                    messages_content = [
                        {"type": "text", "text": prompt0},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image0}", "detail": "high"}},
                        {"type": "text", "text": prompt_image0_append},
                        {"type": "text", "text": prompt_image_current_prepend},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"}},
                    ]
                else:
                    messages_content = [
                        prompt0,
                        {"mime_type": "image/jpeg", "data": base64_image0},
                        prompt_image0_append,
                        prompt_image_current_prepend,
                        {"mime_type": "image/jpeg", "data": base64_image_current},
                    ]
                continue
            else:
                return current_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, base64_image_current

        # Case A: progress estimate
        if "completion percentage: " in response_text:
            if "Frame description: " in response_text:
                current_frame_description = response_text.split("Frame description: ")[1].split("\n")[0]
            else:
                current_frame_description = ""

            progress_str = response_text.split("completion percentage: ")[1].split("%")[0].strip()
            try:
                current_progress = int(progress_str)
            except ValueError:
                current_progress = 0

            frame_description_list.append(current_frame_description)
            progress_list.append(current_progress)

            print("Current Frame Description:", current_frame_description)
            print("Current Progress:", current_progress)

            current_idx += 1

            # Continue frame-by-frame scoring if not finished
            if current_progress < subtask_completion_threshold and current_idx < len(frame_file_list):
                if current_progress != prev_progress:
                    prev_progress = current_progress
                    base64_image_prev = base64_image_current

                base64_image_current = encode_image(frame_file_list[current_idx])

                # Make prompt reduction robust to missing split markers
                if ("IF THE GIVEN SUBTASK" in prompt0) and ("IF THE SUBTASK IS NOT DECOMPOSABLE (SEE EXAMPLES ABOVE), " in prompt0):
                    prompt0_reduced = prompt0.split("IF THE GIVEN SUBTASK")[0] + prompt0.split(
                        "IF THE SUBTASK IS NOT DECOMPOSABLE (SEE EXAMPLES ABOVE), "
                    )[-1]
                else:
                    prompt0_reduced = prompt0

                if "gpt" in model_name:
                    messages_content = [
                        {"type": "text", "text": prompt0_reduced},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image0}", "detail": "high"}},
                        {"type": "text", "text": prompt_image0_append},
                        {"type": "text", "text": prompt_image_prev_prepend},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_prev}", "detail": "high"}},
                        {"type": "text", "text": prompt_image_prev_append_template.format(prev_progress=prev_progress)},
                        {"type": "text", "text": prompt_image_current_prepend},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"}},
                    ]
                else:
                    messages_content = [
                        prompt0_reduced,
                        {"mime_type": "image/jpeg", "data": base64_image0},
                        prompt_image0_append,
                        prompt_image_prev_prepend,
                        {"mime_type": "image/jpeg", "data": base64_image_prev},
                        prompt_image_prev_append_template.format(prev_progress=prev_progress),
                        prompt_image_current_prepend,
                        {"mime_type": "image/jpeg", "data": base64_image_current},
                    ]
            else:
                print("Progress List:", progress_list)
                return current_idx, subtask_list, progress_list, frame_description_list, base64_image_current

        # Case B: decomposition request
        elif "robot needs to" in response_text:
            # Ensure the container structure exists even if "New subtasks:" is missing
            if len(subtask_list) == 0:
                if "New subtasks: " in response_text:
                    raw = (
                        response_text.split("New subtasks: ")[1]
                        .split("\n")[0]
                        .replace("'", "")
                        .replace("[", "")
                        .replace("]", "")
                        .strip()
                    )
                    items = [x for x in raw.split(", ") if x]
                    new_subtask_list = [[x] for x in items]
                else:
                    new_subtask_list = []

                subtask_list = [new_subtask_list]
                subtask_progress_list = [[[] for _ in new_subtask_list]]
                subtask_frame_descriptions_list = [[[] for _ in new_subtask_list]]

            try:
                new_subtask = response_text.split("robot needs to: ")[1].split("\n")[0].split(".")[0].lower()
            except Exception:
                new_subtask = "unknown subtask"

            subtask_idx += 1
            if subtask_idx >= len(subtask_list[0]):
                subtask_list[0].append([new_subtask])
                subtask_progress_list[0].append([])
                subtask_frame_descriptions_list[0].append([])

            # Critical fix: recursion must pass model_name and clients correctly, and keep argument order consistent
            (
                current_idx,
                new_subtask_list2,
                new_subtask_progress_list2,
                new_subtask_frame_descriptions_list2,
                new_base64_image_current2,
            ) = rover(
                model_name=model_name,
                task_description_i=new_subtask,
                camera_view=camera_view,
                frame_file_list=frame_file_list,
                current_start_idx=current_idx - 1,
                subtask_completion_threshold=subtask_completion_threshold,
                include_last_subtask_frame=include_last_subtask_frame,
                openai_client=openai_client,
                gemini_model=gemini_model,
            )

            subtask_list[0][subtask_idx].append(new_subtask_list2)
            subtask_progress_list[0][subtask_idx] = new_subtask_progress_list2
            subtask_frame_descriptions_list[0][subtask_idx] = new_subtask_frame_descriptions_list2

            # Build the next "state" prompt
            if current_idx < len(frame_file_list):
                if "gpt" in model_name:
                    messages_content = [{"type": "text", "text": task_prepend}]
                    for t in response_text_list:
                        messages_content.append(
                            {"type": "text", "text": t.replace("The robot needs to: ", "The robot has completed: ")}
                        )
                    if include_last_subtask_frame and new_base64_image_current2 is not None:
                        messages_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{new_base64_image_current2}", "detail": "high"}}
                        )
                else:
                    messages_content = [task_prepend]
                    for t in response_text_list:
                        messages_content.append(t.replace("The robot needs to: ", "The robot has completed: "))
                    if include_last_subtask_frame and new_base64_image_current2 is not None:
                        messages_content.append({"mime_type": "image/jpeg", "data": new_base64_image_current2})
            else:
                return current_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, base64_image_current

        # Case C: task done
        elif "task complete" in response_text.lower():
            print("Task complete")
            return current_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, base64_image_current


# -----------------------------
# Post-processing
# -----------------------------
def process_rover_output(subtask_list, subtask_progress_list, subtask_frame_descriptions_list, frame_file_list):
    """
    Convert the nested decomposition output into per-frame progress and descriptions.
    frame_file_list must be passed in; the original code referenced it as an undefined global.
    """
    final_progress_list = depth_first_traverse_rover_output_structure(
        subtask_list[0], subtask_progress_list[0], len(subtask_list[0])
    )

    while len(final_progress_list) < len(frame_file_list) - 1:
        final_progress_list.append(final_progress_list[-1])

    frame_descriptions_list = list(flatten(subtask_frame_descriptions_list))
    while len(frame_descriptions_list) < len(frame_file_list) - 1:
        frame_descriptions_list.append(frame_descriptions_list[-1] if frame_descriptions_list else "")

    return final_progress_list, frame_descriptions_list


def depth_first_traverse_rover_output_structure(subtask_list_i, subtask_progress_list_i, subtask_count, subtask_subtask_idx=None, layer=0):
    """
    Depth-first traversal that maps nested subtask progress (0-100) into a global 0-100 scale.
    """
    indent_str = "----" * 2 * (layer + 1) + " "
    layer_tag = f"\n{indent_str}layer - {layer}"
    adjusted_progress_list = []

    for subtask_idx in range(len(subtask_list_i)):
        print("Layer Tag:", layer_tag)
        print(indent_str + str(subtask_list_i[subtask_idx][0]))
        print(indent_str + f"subtask_count: {subtask_count}")

        if subtask_subtask_idx is None:
            progress_val_start = (subtask_idx / subtask_count) * 100
        else:
            progress_val_start = (subtask_subtask_idx / subtask_count) * 100

        print(indent_str + f"progress_val_start: {progress_val_start}")

        if len(subtask_list_i[subtask_idx]) > 1:
            if len(subtask_list_i[subtask_idx][1]) > 0:
                subtask_subtask_count = len(subtask_list_i[subtask_idx][1][0])
                for i in range(subtask_subtask_count):
                    if subtask_subtask_idx is None:
                        adjusted_progress_list2 = depth_first_traverse_rover_output_structure(
                            [subtask_list_i[subtask_idx][1][0][i]],
                            subtask_progress_list_i[subtask_idx][0],
                            subtask_subtask_count,
                            subtask_subtask_idx=i,
                            layer=layer + 1,
                        )
                    else:
                        adjusted_progress_list2 = depth_first_traverse_rover_output_structure(
                            [subtask_list_i[subtask_idx][1][0][i]],
                            subtask_progress_list_i[subtask_subtask_idx][0],
                            subtask_subtask_count,
                            subtask_subtask_idx=i,
                            layer=layer + 1,
                        )

                    adjusted_progress_list += [progress_val_start + (x / subtask_count) for x in adjusted_progress_list2]
            else:
                print(indent_str + "raw progress values for this subtask")
                if subtask_subtask_idx is None:
                    print(indent_str + str(subtask_progress_list_i[subtask_idx]))
                    adjusted_progress_list += [progress_val_start + (x / subtask_count) for x in subtask_progress_list_i[subtask_idx]]
                else:
                    print(indent_str + str(subtask_progress_list_i[subtask_subtask_idx]))
                    adjusted_progress_list += [progress_val_start + (x / subtask_count) for x in subtask_progress_list_i[subtask_subtask_idx]]
        else:
            print(indent_str + "raw progress values for this subtask")
            print(indent_str + str([]))

    print(indent_str + "scaled progress values for this subtask")
    print(indent_str + str(adjusted_progress_list))
    return adjusted_progress_list
