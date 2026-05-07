import argparse
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
import torch.nn.functional as F
import random
from torch.utils.data import Dataset

class TripletFaceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels_to_indices = {}
        self.valid_indices = []  # Only indices of samples from valid classes
        
        # Group indices by label
        for idx, (_, label) in enumerate(dataset.imgs):
            if label not in self.labels_to_indices:
                self.labels_to_indices[label] = []
            self.labels_to_indices[label].append(idx)
        
        # Filter out classes with less than 2 samples and build valid indices list
        for label, indices in self.labels_to_indices.items():
            if len(indices) >= 2:
                self.valid_indices.extend(indices)
        
        self.labels_to_indices = {label: indices for label, indices in self.labels_to_indices.items() if len(indices) >= 2}
        self.labels = list(self.labels_to_indices.keys())
        
        # Create a mapping from original labels to new continuous labels [0, 1, 2, ...]
        self.label_mapping = {original_label: new_label for new_label, original_label in enumerate(self.labels)}
    
    def __getitem__(self, index):
        # Map index to actual dataset index
        actual_index = self.valid_indices[index]
        
        # Get anchor sample
        anchor_path, anchor_label = self.dataset.imgs[actual_index]
        anchor_img, _ = self.dataset[actual_index]
        
        # Convert to mapped label for training (continuous from 0 to N-1)
        mapped_anchor_label = self.label_mapping[anchor_label]
        
        # Get positive sample (same class, different image)
        positive_indices = self.labels_to_indices[anchor_label]
        # Make sure we have other samples in the same class
        available_indices = [idx for idx in positive_indices if idx != actual_index]
        if len(available_indices) > 0:
            positive_index = random.choice(available_indices)
        else:
            # If no other samples, use the same sample (fallback)
            positive_index = actual_index
        positive_img, _ = self.dataset[positive_index]
        
        # Get negative sample (different class)
        negative_labels = [label for label in self.labels if label != anchor_label]
        if len(negative_labels) > 0:
            negative_label = random.choice(negative_labels)
            negative_index = random.choice(self.labels_to_indices[negative_label])
        else:
            # Fallback if there's only one class
            negative_index = random.choice(self.valid_indices)
            while negative_index == actual_index:
                negative_index = random.choice(self.valid_indices)
            negative_label = self.dataset.imgs[negative_index][1]
        negative_img, _ = self.dataset[negative_index]
        
        return anchor_img, positive_img, negative_img, mapped_anchor_label
    
    def __len__(self):
        return len(self.valid_indices)

class RemapLabelsDataset(Dataset):
    """Wrapper dataset to remap labels to continuous range [0, N-1]"""
    def __init__(self, dataset, label_mapping):
        self.dataset = dataset
        self.label_mapping = label_mapping
    
    def __getitem__(self, index):
        img, original_label = self.dataset[index]
        # Remap the label to continuous range
        remapped_label = self.label_mapping[original_label]
        return img, remapped_label
    
    def __len__(self):
        return len(self.dataset)

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        dist_ap = F.pairwise_distance(anchor, positive, keepdim=True)
        dist_an = F.pairwise_distance(anchor, negative, keepdim=True)
        loss = torch.relu(dist_ap - dist_an + self.margin)
        return loss.mean()

class CombinedLoss(torch.nn.Module):
    def __init__(self, margin=0.2, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = TripletLoss(margin)
        self.alpha = alpha
        self.ce_loss = torch.nn.CrossEntropyLoss()
        
    def forward(self, outputs, labels, anchor_embed, positive_embed, negative_embed):
        ce = self.ce_loss(outputs, labels)
        triplet = self.triplet_loss(anchor_embed, positive_embed, negative_embed)
        return self.alpha * ce + (1 - self.alpha) * triplet

def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Recognition Model')
    parser.add_argument('--data_dir', type=str, default='/home/huachenghao/codes/face_index-160', 
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to train on (cuda:n or cpu)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--pretrained_weights', type=str, default='./model_data/20180408-102900-casia-webface.pt',
                        help='Path to pretrained weights')
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for triplet loss')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for combining CE and triplet loss')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    workers = 0 if os.name == 'nt' else 8

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Prepare transformations
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=trans)
    num_classes = len(dataset.classes)
    print(f'Dataset loaded with {len(dataset)} images and {num_classes} classes')
    
    # Check if we have enough classes
    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes for classification, but got {num_classes}")
    
    # Create triplet dataset
    triplet_dataset = TripletFaceDataset(dataset)
    actual_num_classes = len(triplet_dataset.labels)
    print(f'Filtered dataset has {len(triplet_dataset)} images and {actual_num_classes} classes with 2+ samples each')
    
    # Split dataset into train and validation sets based on valid indices
    valid_img_inds = np.array(triplet_dataset.valid_indices)
    np.random.shuffle(valid_img_inds)
    split_point = int(0.8 * len(valid_img_inds))
    train_inds = valid_img_inds[:split_point]
    val_inds = valid_img_inds[split_point:]
    
    # Create mappings from global indices to local indices for the triplet dataset
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(triplet_dataset.valid_indices)}
    train_local_inds = [global_to_local[idx] for idx in train_inds]
    val_local_inds = [global_to_local[idx] for idx in val_inds]

    train_loader = DataLoader(
        triplet_dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_local_inds)
    )
    
    # For validation, we need to create a subset of the original dataset with remapped labels
    val_subset = torch.utils.data.Subset(dataset, val_inds)
    remapped_val_dataset = RemapLabelsDataset(val_subset, triplet_dataset.label_mapping)
    val_loader = DataLoader(
        remapped_val_dataset,
        num_workers=workers,
        batch_size=batch_size
    )

    # Initialize model with pretrained weights
    print(f"Loading pretrained weights from: {args.pretrained_weights}")
    # Initialize model in feature extraction mode first to get embeddings
    resnet_feature = InceptionResnetV1(pretrained=None, classify=False)
    
    # Then create a model for classification
    resnet = InceptionResnetV1(pretrained=None, classify=True, num_classes=actual_num_classes)
    
    # Load the pretrained weights
    pretrained_dict = torch.load(args.pretrained_weights, map_location=device)
    
    # Load weights into feature extractor
    resnet_feature.load_state_dict(pretrained_dict, strict=False)
    
    # Copy relevant weights to classification model
    model_dict = resnet.state_dict()
    pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'logits' not in k}
    model_dict.update(pretrained_dict_filtered)
    resnet.load_state_dict(model_dict)
    
    print("Pretrained weights loaded successfully")
    
    resnet = resnet.to(device)
    resnet_feature = resnet_feature.to(device)  # Keep feature extractor for triplet loss

    # Setup optimizer and scheduler
    optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, [15, 10])

    # Loss function
    loss_fn = CombinedLoss(margin=args.margin, alpha=args.alpha)
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    # TensorBoard writer
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial validation')
    print('-' * 20)
    resnet.eval()
    # For validation, we only use CE loss
    val_loss_fn = torch.nn.CrossEntropyLoss()
    training.pass_epoch(
        resnet, val_loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    # Training loop
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 20)

        # Train phase
        resnet.train()
        train_losses = []
        for batch_idx, (anchor_img, positive_img, negative_img, labels) in enumerate(train_loader):
            anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass for all three images through feature extractor to get embeddings
            with torch.no_grad():
                anchor_embed_feat = resnet_feature(anchor_img)
                positive_embed_feat = resnet_feature(positive_img)
                negative_embed_feat = resnet_feature(negative_img)
            
            # Forward pass for anchor image through classification model
            outputs = resnet(anchor_img)
            
            # Calculate combined loss
            loss = loss_fn(outputs, labels, anchor_embed_feat, positive_embed_feat, negative_embed_feat)
            train_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log metrics
            if batch_idx % writer.interval == 0:
                acc = training.accuracy(outputs, labels).item()  # Fixed: use .item() instead of indexing
                writer.add_scalar('Loss/Train', loss.item(), writer.iteration)
                writer.add_scalar('Accuracy/Train', acc, writer.iteration)
                writer.iteration += 1
                
        scheduler.step()
        
        # Validation phase
        resnet.eval()
        val_results = training.pass_epoch(
            resnet, val_loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )
        
        print(f'Train Loss: {np.mean(train_losses):.4f}')
        
        # Save model checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'label_mapping': triplet_dataset.label_mapping  # Save label mapping for inference
        }, checkpoint_path)
        print(f'Model saved to {checkpoint_path}')

    writer.close()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(resnet.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    
    # Save class to index mapping
    class_to_idx_path = os.path.join(args.output_dir, 'class_to_idx.npy')
    # Save the mapping with remapped labels
    remapped_class_to_idx = {class_name: triplet_dataset.label_mapping[original_idx] 
                            for class_name, original_idx in dataset.class_to_idx.items() 
                            if original_idx in triplet_dataset.label_mapping}
    np.save(class_to_idx_path, remapped_class_to_idx)
    print(f'Class to index mapping saved to {class_to_idx_path}')

if __name__ == '__main__':
    main()