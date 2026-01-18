import torch
import cv2
import numpy as np
import argparse
import os
from torchvision import transforms
from models import CNNLSTM, C3D_ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_saliency_map(model, input_tensor):
    model.eval()
    input_tensor.requires_grad = True
    output = model(input_tensor)
    
    # Target "Violence" class
    score = output[0, 1] 
    score.backward()
    
    # Get gradients
    gradients = input_tensor.grad.data 
    saliency = gradients.abs().max(dim=2)[0].squeeze(0) # (Frames, H, W)
    return saliency

def process_heatmap(heatmap_low_res, target_size):
    """
    NUCLEAR OPTION: Heavy processing to merge dots into blobs.
    """
    h, w = target_size
    heatmap = heatmap_low_res.cpu().numpy()
    
    # 1. Threshold: Ignore weak noise (bottom 30%)
    heatmap[heatmap < np.mean(heatmap) + 0.5 * np.std(heatmap)] = 0
    
    # 2. Normalize to 0-255 for OpenCV processing
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = np.uint8(255 * heatmap)
    
    # 3. DILATION (The Fix): Make dots fatter so they merge
    kernel_dilate = np.ones((5, 5), np.uint8) 
    heatmap = cv2.dilate(heatmap, kernel_dilate, iterations=2)
    
    # 4. CLOSING: Fill small holes between dots
    kernel_close = np.ones((9, 9), np.uint8)
    heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_CLOSE, kernel_close)
    
    # 5. Heavy Blur to make it smooth
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    # 6. Resize to HD
    heatmap_hd = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 7. Apply Color Map
    heatmap_color = cv2.applyColorMap(heatmap_hd, cv2.COLORMAP_JET)
    
    return heatmap_color

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='A', help='Model type: A or B')
    parser.add_argument('--video', type=str, required=True, help='Path to video')
    args = parser.parse_args()

    # Load Model
    print(f"Loading Model {args.model}...")
    if args.model == 'A':
        model = CNNLSTM().to(device)
        path = 'violence_model_A.pth'
    else:
        model = C3D_ResNet().to(device)
        path = 'violence_model_B.pth'
        
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    model.load_state_dict(torch.load(path, map_location=device))
    
    # Process Video
    cap = cv2.VideoCapture(args.video)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_for_model = []
    frames_high_res = []
    
    print("Reading video...")
    while len(frames_for_model) < 16:
        ret, frame = cap.read()
        if not ret: break
        
        frames_high_res.append(frame)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (128, 128))
        frames_for_model.append(frame_rgb)
        
    cap.release()
    
    if len(frames_for_model) < 16:
        print("Video too short!")
        return

    # Tensor Prep
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = torch.stack([transform(f) for f in frames_for_model]).unsqueeze(0).to(device)

    # Compute Saliency
    print("Computing Saliency...")
    saliency_small = get_saliency_map(model, input_tensor)
    
    # Render
    out_name = f"explained_FINAL_{args.model}.avi"
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'DIVX'), 5, (orig_w, orig_h))
    
    print("Rendering Final Heatmap...")
    for i in range(len(frames_high_res)):
        # Process the heatmap with Dilation/Blur
        heatmap_overlay = process_heatmap(saliency_small[i], (orig_h, orig_w))
        
        # Overlay
        final_frame = cv2.addWeighted(frames_high_res[i], 0.7, heatmap_overlay, 0.3, 0)
        out.write(final_frame)
        
    out.release()
    print(f"SUCCESS! Saved: {out_name}")

if __name__ == "__main__":
    main()