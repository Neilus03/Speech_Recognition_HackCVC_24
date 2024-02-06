from typing import Optional, Union, Tuple, List
from pathlib import Path
import argparse
import yaml
import numpy as np
import textwrap
import random
import string
import cv2
import requests
from io import BytesIO
from PIL import Image



CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Demo:

    def __init__(
        self,
        frame_buffer_size: int,
        api_keypoints: str,
        text_rel_pos_y: float = 0.8,
        text_rel_pos_x: float = 0.5,
        window_width: Optional[int] = None,
        window_height: Optional[int] = None,
        max_line_len: Optional[int] = None,
        window_title: str = "Speech Recognition"
    ):
        self.api_keypoints = api_keypoints
        self.frame_buffer_size = frame_buffer_size
        self.text_rel_pos_y = text_rel_pos_y
        self.text_rel_pos_x = text_rel_pos_x
        self.max_line_len = max_line_len
        self.frame_buffer = []
        self.window_width = window_width
        self.window_height = window_height
        self.window_title = window_title
        self.current_text = ""


    def _to_keypoints(self, frames: list):
        try:
            files = []
            for frame in frames:
                image_pil = Image.fromarray(frames[0])
                image_bytes_io = BytesIO()
                image_pil.save(image_bytes_io, format='PNG')
                files.append(('files', ('image', image_bytes_io.getvalue(), 'image/jpeg')))
            response = requests.post(self.api_keypoints, files=files)
            data = response.json()
            return data
        except Exception as e:
            print(e)
            return None

    def _to_lip_model(self, frames: list):
        # TODO: call lip model
        text = "".join(random.choice(string.ascii_lowercase) for i in range(random.randint(30,70)))
        return text

    def _process_frame(self, frame: np.array):
        # Get keypoints
        keypoints = self._to_keypoints([frame])
        if keypoints and keypoints[0] and keypoints[0]:
            frame = self._draw_keypoints(keypoints[0][0], frame)
            # Add frame to buffer and call the lip reading model
            if len(self.frame_buffer) < self.frame_buffer_size:
                self.frame_buffer.append((frame, keypoints[0][0]))
            else:
                self.current_text = self._to_lip_model(self.frame_buffer)
                self.frame_buffer = []

        # Resize the video
        frame = self._resize_frame(frame)

        # Add text
        frame = self._insert_text(frame, self.current_text)

        return frame

    def _insert_text(self, frame: np.ndarray, text: str) -> np.ndarray:
        h,w,_ = frame.shape
        text_pos = (int(self.text_rel_pos_x*w), int(self.text_rel_pos_y*h))
        if self.max_line_len:
            text = textwrap.wrap(text, self.max_line_len, break_long_words=True)

        frame = self._draw_text(
            frame,
            text,
            text_pos,
            margin=10,
            center_x=True,
            background_alpha=0.5
        )
        return frame

    def _draw_text(
        self,   
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

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        h,w,_ = frame.shape
        if self.window_height and self.window_width:
            frame = cv2.resize(frame, (self.window_height, self.window_width))
        elif self.window_height:
            fx = self.window_height/h
            frame = cv2.resize(frame, None, fx=fx, fy=fx)
        elif self.window_width:
            fx = self.window_width/w
            frame = cv2.resize(frame, None, fx=fx, fy=fx)
        return frame

    def start_camera(self, device: int = 0):
        self.camera = cv2.VideoCapture(device)
        while(True): 
            ret, frame = self.camera.read()
            frame = self._process_frame(frame)
            cv2.imshow(self.window_title, frame) 
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break

    def _draw_keypoints(self, keypoints: dict, frame: np.ndarray) -> np.ndarray:
        points = keypoints["top_lip"] + keypoints["bottom_lip"]
        for point in points:
            frame = cv2.circle(frame, point, 3, (255, 0, 0), -1)
        return frame




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
    )
    ap.add_argument(
        "--width",
        help="Window width",
        type=int
    )
    ap.add_argument(
        "--height",
        help="Window height",
        type=int
    )
    args = ap.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    demo_args = config["demo"]
    demo = Demo(
        **demo_args,
        window_height=args.height,
        window_width=args.width,
    )
    demo.start_camera()
