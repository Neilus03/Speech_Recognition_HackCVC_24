from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
import argparse
import textwrap
import string
import random

from demo_video import create_first_graph, create_second_graph, inference_video


CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_REL_POS_X = 0.5
TEXT_REL_POS_Y = 0.9
MAX_LINE_LEN = 100
TEXT_SPLITS = 4



def create_video(input_video_path: str, output_video_path: str, fps: int = 25):
    video_in = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_out = None

    if not video_in.isOpened():
        print(f"Error: Could not open the video file at {input_video_path}")
        return

    num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

    # TODO:
    text = Path("/home/GROUP02/videos/carles_whisper.txt").read_text()
    texts = textwrap.wrap(text, num_frames, break_long_words=True)

    while True:
        ret, frame = video_in.read()
        if not ret:
            break
        if video_out is None:
            h,w,_ = frame.shape
            video_out = cv2.VideoWriter(output_video_path, fourcc, 25, (w,h))

        # TODO: Plot things
        # first_graph, second_graph, face_positions = inference_video(frame)
        

        frame = insert_text(frame, text)

        video_out.write(frame)
        print(".", end="", flush=True)

    video_in.release()
    video_out.release()
    print("")


def insert_text(frame: np.ndarray, text: str) -> np.ndarray:
    h,w,_ = frame.shape
    text_pos = (int(TEXT_REL_POS_X*w), int(TEXT_REL_POS_Y*h))
    if MAX_LINE_LEN:
        text = textwrap.wrap(text, MAX_LINE_LEN, break_long_words=True)

    frame = draw_text(
        frame,
        text,
        text_pos,
        margin=10,
        center_x=True,
        background_alpha=0.5
    )
    return frame

def draw_text(
    image: np.ndarray,
    text: Union[str, List[str]],
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
    thickness: int = 1,
    line_space: int = 15,
    background: bool = True,
    background_color: Tuple[int, int, int] = (50, 50, 50),
    background_alpha: float = 1,
    margin: int = 0,
    center_x: bool = False,
) -> np.ndarray:
    if not text:
        return image

    text_lines = text.splitlines() if isinstance(text, str) else text
    
    # Compute the final text size
    max_text_len = max(text_lines, key=lambda x: len(x))
    (text_w, text_h), _ = cv2.getTextSize(
        max_text_len,
        CV2_FONT,
        scale,
        thickness
    )
    box_h = text_h + ((text_h + scale * line_space) *
                    (len(text_lines) - 1)) + (margin * 2)

    # Compute the text starting position
    x, y = position
    if center_x:
        x = x-text_w//2+margin

    # Clip values
    x = min(max(x, 0), image.shape[1] - 1)
    y = min(max(y, 0), image.shape[0] - 1)

    # Draw background
    if background and background_alpha > 0:
        xmin = max(x, 0)
        ymin = max(y - box_h, 0)
        xmax = min(x + text_w + (margin * 2), image.shape[1])
        ymax = min(y, image.shape[0])
        
        if background_alpha < 1:
            bg = np.full(
                (ymax-ymin, xmax-xmin, 3),
                background_color,
                dtype="uint8"
            )
            image[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(
                image[ymin:ymax, xmin:xmax, :],
                1-background_alpha,
                bg,
                background_alpha,
                0
            )
        else:
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                background_alpha,
                -1
            )

    # Draw the text lines
    for i, line in enumerate(reversed(text_lines)):
        dy = i * (text_h + scale * line_space)
        cv2.putText(
            image, line, (x + margin, y - margin - dy), CV2_FONT, scale, color,
            thickness, cv2.LINE_AA)

    return image


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--video-in",
        required=True
    )
    ap.add_argument(
        "-o",
        "--video-out",
        required=True
    )
    ap.add_argument(
        "--fps",
        default=25
    )
    args = ap.parse_args()
    create_video(args.video_in, args.video_out, args.fps)
