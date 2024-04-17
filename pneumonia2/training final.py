

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from sklearn.metrics import confusion_matrix

cudnn.benchmark = True
plt.ion()  # Interactive mode

transformers = {
    'train_transforms': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test_transforms': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid_transforms': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

trans = list(transformers.keys())

path = 'C:/Users/ABI PRIYANKA/Downloads/pnuemonia'
categories = ['train', 'valid', 'test']

dset = {
    x: datasets.ImageFolder(
        os.path.join(path, x),
        transform=transformers[trans[i]]
    )
    for i, x in enumerate(categories)
}




from sklearn.utils import resample

# Separate data for upsampling and downsampling
train_data = dset['train'].samples.copy()
train_targets = dset['train'].targets.copy()


# Convert train_targets to a numpy array
train_targets = np.array(train_targets)

# Show the number of images in each class
print("Original Train Dataset:")
print("Normal class count:", np.bincount(train_targets.astype(int))[0])
print("Pneumonia class count:", np.bincount(train_targets.astype(int))[1])
print()


# Upsample "NORMAL" class
normal_indices = np.where(train_targets == 0)[0]
upsampled_normal_indices = resample(normal_indices, replace=True, n_samples=len(normal_indices) * 2, random_state=42)

# Downsample "PNEUMONIA" class
pneumonia_indices = np.where(train_targets == 1)[0]
downsampled_pneumonia_indices = resample(pneumonia_indices, replace=False, n_samples=len(pneumonia_indices) // 2, random_state=42)

# Combine the upsampled "NORMAL" class and downsampled "PNEUMONIA" class
resampled_indices = np.concatenate((upsampled_normal_indices, downsampled_pneumonia_indices))

# Create a new train dataset with balanced classes
resampled_data = [train_data[i] for i in resampled_indices]
resampled_targets = train_targets[resampled_indices]



print("Resampled Train Dataset:")
print("Normal class count:", np.bincount(resampled_targets.astype(int))[0])
print("Pneumonia class count:", np.bincount(resampled_targets.astype(int))[1])


dataset_sizes = {x: len(dset[x]) for x in categories}

# Update the 'train' dataset with resampled data
dset['train'].samples = resampled_data
dset['train'].targets = resampled_targets
dataset_sizes['train'] = len(dset['train'])


num_threads = 4

dataloaders = {
    x: torch.utils.data.DataLoader(
        dset[x], batch_size=64, shuffle=True, num_workers=num_threads
    )
    for x in categories
}
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
inputs, classes = next(iter(dataloaders["train"]))
out = torchvision.utils.make_grid(inputs)
class_names = dset["train"].classes
imshow(out, title=[class_names[x] for x in classes])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    true_labels = []
    predicted_labels = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history if only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                true_labels.extend(labels.data.tolist())
                predicted_labels.extend(preds.tolist())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                valid_losses.append(epoch_loss)
                valid_accs.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copy the model if it's the best so far
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'good_model.pth')


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    class_names = ['Normal', 'Pneumonia']
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    return model, train_losses, valid_losses, train_accs, valid_accs

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
model, train_losses, valid_losses, train_accs, valid_accs = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=5)





# Save the model with the best validation accuracy
best_model_filename = 'good_model.pth'
torch.save(model.state_dict(), best_model_filename)
print("Best model saved as:", best_model_filename)





