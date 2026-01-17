# Violence Detection Project

## 1. Setup
1. Download the RWF-2000 dataset from Kaggle or GitHub.
2. Extract it into the `data/` folder.
   It should look like this:
   ViolenceDetection/data/RWF-2000/train/Fight/
   ViolenceDetection/data/RWF-2000/train/NonFight/
   
3. Install libraries:
   pip install -r requirements.txt

## 2. Training
# Train Model A (CNN+LSTM):
python train.py --model A --epochs 10

# Train Model B (3D-CNN):
python train.py --model B --epochs 10
