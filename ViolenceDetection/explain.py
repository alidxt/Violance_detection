import torch
import cv2
import numpy as np
import argparse
from torchvision import transforms
from models import CNNLSTM, C3D_ResNet
from dataset import RWFDataset

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_saliency_map(model, input_tensor):
    """
    Computes gradients with respect to input video to find 'important' pixels.
    """
    model.eval()
    
    # We need gradients for the input image
    input_tensor.requires_grad = True
    
    # Forward pass
    output = model(input_tensor)
    
    # We want to explain the 'Violence' class (index 1)
    # Score for class 1
    score = output[0, 1] 
    
    # Backward pass to get gradients
    score.backward()
    
    # Get gradients: (Batch, Frames, Channels, H, W)
    gradients = input_tensor.grad.data
    
    # Calculate magnitude of gradients (Saliency)
    # Max across channels (R,G,B) -> (Batch, Frames, H, W)
    saliency = gradients.abs().max(dim=2)[0] 
    
    # Take the max value across the batch (since batch=1)
    saliency = saliency.squeeze(0) # (Frames, H, W)
    
    return saliency

def overlay_heatmap(original_frame, saliency_map):
    """
    Overlays the saliency heatmap on the original frame.
    """
    # Normalize saliency to 0-255
    heatmap = saliency_map.cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-20) # Avoid div by zero
    heatmap = np.uint8(255 * heatmap)
    
    # Apply color map (JET is standard for heatmaps)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Combine (Original 60% + Heatmap 40%)
    overlay = cv2.addWeighted(original_frame, 0.6, heatmap_color, 0.4, 0)
    return overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--model_type', type=str, default='A', help='A or B')
    parser.add_argument('--video_path', type=str, required=True, help='Path to a test video')
    args = parser.parse_args()

    # 1. Load Model
    print("Loading Model...")
    if args.model_type == 'A':
        model = CNNLSTM().to(device)
    else:
        model = C3D_ResNet().to(device)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 2. Load Video
    print("Processing Video...")
    cap = cv2.VideoCapture(args.video_path)
    frames = []
    originals = []
    
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        # Save original for visualization
        orig = cv2.resize(frame, (128, 128))
        originals.append(orig)
        # Preprocess for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (128, 128))
        frames.append(frame_rgb)
        
    cap.release()
    
    # Convert to Tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = torch.stack([transform(f) for f in frames])
    input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dim -> (1, 16, 3, 128, 128)

    # 3. Compute Saliency
    print("Computing Saliency Map...")
    saliency_maps = get_saliency_map(model, input_tensor)
    
    # 4. Save Output Video
    out_path = "explained_video.avi"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 5, (128, 128))
    
    for i in range(len(originals)):
        # Get frame and its corresponding saliency
        frame = originals[i]
        saliency = saliency_maps[i]
        
        # Create overlay
        result = overlay_heatmap(frame, saliency)
        out.write(result)
        
    out.release()
    print(f"Done! Check {out_path} to see what the model is looking at.")

if __name__ == "__main__":
    main()