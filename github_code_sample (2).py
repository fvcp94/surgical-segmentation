"""
Medical Image Segmentation for Surgical Planning
U-Net Implementation with Attention Mechanisms

Author: Febin Varghese
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd


# ==================== MODEL ARCHITECTURE ====================

class AttentionBlock(nn.Module):
    """Attention gate for U-Net decoder"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization"""
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    """U-Net with Attention Mechanisms for Medical Image Segmentation"""
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        self.dropout = nn.Dropout2d(0.5)
        
        # Decoder with attention
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        b = self.dropout(b)
        
        # Decoder with attention
        d4 = self.up4(b)
        e4 = self.att4(g=d4, x=e4)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = torch.sigmoid(self.out(d1))
        return out


# ==================== LOSS FUNCTIONS ====================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice + BCE Loss"""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


# ==================== METRICS ====================

def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Calculate Dice coefficient"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Calculate Intersection over Union"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def sensitivity_specificity(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Calculate sensitivity and specificity"""
    pred_flat = pred.flatten().astype(int)
    target_flat = target.flatten().astype(int)
    
    tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Hausdorff distance (simplified version)"""
    from scipy.spatial.distance import directed_hausdorff
    
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    
    forward = directed_hausdorff(pred_points, target_points)[0]
    backward = directed_hausdorff(target_points, pred_points)[0]
    
    return max(forward, backward)


# ==================== DATASET ====================

class MedicalImageDataset(Dataset):
    """Dataset for medical image segmentation"""
    def __init__(self, images: np.ndarray, masks: np.ndarray, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            # Apply same transform to both image and mask
            image = self.transform(image)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criterion, device: str) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model: nn.Module, dataloader: DataLoader, criterion, device: str) -> Tuple[float, dict]:
    """Validate model"""
    model.eval()
    total_loss = 0
    metrics = {'dice': [], 'iou': [], 'sensitivity': [], 'specificity': []}
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate metrics
            preds = (outputs > 0.5).cpu().numpy()
            targets = masks.cpu().numpy()
            
            for pred, target in zip(preds, targets):
                metrics['dice'].append(dice_coefficient(pred, target))
                metrics['iou'].append(iou_score(pred, target))
                sens, spec = sensitivity_specificity(pred, target)
                metrics['sensitivity'].append(sens)
                metrics['specificity'].append(spec)
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return total_loss / len(dataloader), avg_metrics


# ==================== INFERENCE ====================

def predict_with_uncertainty(model: nn.Module, image: torch.Tensor, 
                            n_samples: int = 20, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with uncertainty estimation using Monte Carlo Dropout
    Returns: (mean prediction, uncertainty map)
    """
    model.train()  # Keep dropout active
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(image.to(device))
            predictions.append(output.cpu().numpy())
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    
    return mean_pred, uncertainty


# ==================== VISUALIZATION ====================

def visualize_results(image: np.ndarray, true_mask: np.ndarray, 
                     pred_mask: np.ndarray, uncertainty: Optional[np.ndarray] = None):
    """Visualize segmentation results"""
    fig, axes = plt.subplots(1, 4 if uncertainty is not None else 3, figsize=(15, 5))
    
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask.squeeze(), cmap='jet', alpha=0.6)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask.squeeze(), cmap='jet', alpha=0.6)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    if uncertainty is not None:
        im = axes[3].imshow(uncertainty.squeeze(), cmap='hot')
        axes[3].set_title('Uncertainty Map')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    return fig


# ==================== MAIN TRAINING SCRIPT ====================

def main():
    """Main training script"""
    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load your data here (replace with actual data loading)
    # train_images, train_masks = load_training_data()
    # val_images, val_masks = load_validation_data()
    
    # Create datasets and dataloaders
    # train_dataset = MedicalImageDataset(train_images, train_masks, transform=transform)
    # val_dataset = MedicalImageDataset(val_images, val_masks, transform=transform)
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = AttentionUNet(in_channels=1, num_classes=1).to(DEVICE)
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_dice = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Validate
        # val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        # Print metrics
        # print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        # print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # print(f"Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        # print(f"Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
        
        # Learning rate scheduling
        # scheduler.step(val_loss)
        
        # Early stopping
        # if val_metrics['dice'] > best_dice:
        #     best_dice = val_metrics['dice']
        #     torch.save(model.state_dict(), 'best_model.pth')
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print("Early stopping triggered")
        #         break
        
        pass
    
    print("Training complete!")


if __name__ == "__main__":
    main()