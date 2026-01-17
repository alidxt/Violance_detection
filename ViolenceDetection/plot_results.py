import matplotlib.pyplot as plt
import numpy as np

def plot_training_history():
    # -----------------------------------------------------------
    # PASTE YOUR TERMINAL NUMBERS HERE (Example Data)
    # -----------------------------------------------------------
    epochs = range(1, 16) # 1 to 15
    
    # Model A (LSTM) Example Data (Replace with your real results!)
    model_a_train_acc = [50, 55, 62, 68, 72, 75, 78, 80, 82, 83, 85, 86, 87, 88, 88]
    model_a_val_acc   = [48, 52, 60, 65, 68, 70, 72, 74, 73, 75, 76, 76, 77, 78, 77] # Note the overfitting dip
    
    # Model B (3D-CNN) Example Data
    model_b_train_acc = [50, 52, 58, 64, 70, 76, 80, 84, 86, 89, 91, 93, 94, 95, 96]
    model_b_val_acc   = [50, 51, 55, 62, 68, 74, 78, 80, 82, 83, 84, 85, 85, 84, 83]

    # -----------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------
    plt.figure(figsize=(12, 5))

    # Plot 1: Accuracy Comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs, model_a_val_acc, 'b--o', label='Model A (LSTM) Val')
    plt.plot(epochs, model_b_val_acc, 'r-o', label='Model B (3D-CNN) Val')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Overfitting Check (Model B)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, model_b_train_acc, 'r--', label='Training Acc')
    plt.plot(epochs, model_b_val_acc, 'r-', label='Validation Acc')
    plt.title('Model B Overfitting Check')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Save chart
    plt.tight_layout()
    plt.savefig('comparison_curves.png')
    print("Graph saved as comparison_curves.png")
    plt.show()

if __name__ == "__main__":
    plot_training_history()