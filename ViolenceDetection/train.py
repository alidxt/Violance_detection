import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import RWFDataset
from models import CNNLSTM, C3D_ResNet
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='A', help='Model type: A (LSTM) or B (3DCNN)')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    print("Loading Data...")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if data exists
    if not os.path.exists('./data/RWF-2000'):
        print("ERROR: RWF-2000 dataset not found in ./data/ folder.")
        return

    train_set = RWFDataset(root_dir='./data/RWF-2000', phase='train', transform=transform)
    val_set = RWFDataset(root_dir='./data/RWF-2000', phase='val', transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2)

    if args.model == 'A':
        print("Initializing Model A: CNN + LSTM")
        model = CNNLSTM().to(device)
    else:
        print("Initializing Model B: 3D-CNN (ResNet3D)")
        model = C3D_ResNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):    
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), f"violence_model_{args.model}.pth")
    print("Training Complete. Model saved.")

if __name__ == '__main__':
    main()
