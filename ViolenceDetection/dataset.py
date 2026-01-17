import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class RWFDataset(Dataset):
    def __init__(self, root_dir, phase='train', clip_len=16, transform=None):
        self.root_dir = os.path.join(root_dir, phase)
        self.clip_len = clip_len
        self.transform = transform
        
        self.samples = []
        classes = ['NonFight', 'Fight'] 
        
        for label_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.endswith(('.avi', '.mp4')):
                    self.samples.append((os.path.join(class_dir, filename), label_idx))

        print(f"Found {len(self.samples)} videos for {phase}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.load_video(video_path)
        
        # Transform frames
        if self.transform:
            try:
                frames = torch.stack([self.transform(f) for f in frames])
            except Exception as e:
                # Fallback for very weird errors
                print(f"Error transforming video {video_path}: {e}")
                frames = torch.zeros((self.clip_len, 3, 128, 128))

        return frames, label

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- FIX: Handle Corrupt/Empty Videos ---
        if total_frames <= 0:
            print(f"WARNING: Corrupt video found (0 frames): {path}")
            # Return black frames to avoid crash
            return [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(self.clip_len)]
        
        frames = []
        
        # Sampling Strategy
        if total_frames > self.clip_len:
            frame_indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)
        else:
            frame_indices = np.arange(total_frames)
            # Pad if video is too short
            while len(frame_indices) < self.clip_len:
                # Use the last valid frame index to pad
                last_idx = frame_indices[-1] if len(frame_indices) > 0 else 0
                frame_indices = np.append(frame_indices, last_idx)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (128, 128)) 
                    frames.append(frame)
                except Exception:
                    # If resizing fails, skip frame
                    continue
        
        cap.release()
        
        # Final Safety Check: If we didn't get enough frames (read error)
        while len(frames) < self.clip_len:
            frames.append(np.zeros((128, 128, 3), dtype=np.uint8))
            
        return frames[:self.clip_len]