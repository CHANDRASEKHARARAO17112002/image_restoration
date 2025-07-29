from django.shortcuts import render

# Create your views here.

from django.shortcuts import render,redirect
from django.contrib import messages
import urllib.request
import urllib.parse

from django.conf import settings
from django.contrib.auth import authenticate, login

def adminlogin(req):
    if req.method == 'POST':
        username = req.POST.get('username')
        password = req.POST.get('password')
        print("hello")
        print(username,password)
        # Check if the provided credentials match
        if username == 'admin' and password   == 'admin':
            messages.success(req, 'You are logged in.')
            return redirect('admindashboard')  # Redirect to the admin dashboard page
        else:
             messages.error(req, 'You are trying to log in with wrong details.')
             return redirect('admindashboard')  # Redirect to the login page (named 'admin' here)

    # Render the login page if the request method is GET
    return render(req, 'admin/adminlogin.html')

def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('adminlogin')
def admindashboard(req):
    return render(req,'admin/admindashboard.html')
def upload(request):
    return render(request, 'admin/upload.html')

def rf(req):
    return render(req,'admin/rf.html')


# def lenet(req):
#     return render(req,'admin/lenet.html')

import os
import numpy as np
import cv2
import tensorflow as tf
from django.shortcuts import render
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

# Configurations
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 8
EPOCHS = 10

ground_truth_dir = r"E:\dataset\archive (1)\Ground_truth"
noisy_dir = r"E:\dataset\archive (1)\Noisy_folder"

def load_images(path):
    images = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        images.append(img)
    return np.array(images)

X_train_noisy = load_images(noisy_dir)
X_train_clean = load_images(ground_truth_dir)

def build_cnn():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)
    return Model(inputs, x)

def build_generator():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)
    return Model(inputs, x)

def build_discriminator():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, (3, 3), padding="same", activation="sigmoid")(x)
    return Model(inputs, x)

def gan_cnn(request):
    cnn = build_cnn()
    cnn.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    cnn.fit(X_train_noisy, X_train_clean, epochs=10, batch_size=8)
    cnn.save("cnn_denoising_model.h5")

    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer=Adam(0.0002), loss=BinaryCrossentropy())
    
    g_losses, d_losses = [], []
    for epoch in range(EPOCHS):
        idx = np.random.randint(0, X_train_noisy.shape[0], BATCH_SIZE)
        noisy_images = X_train_noisy[idx]
        clean_images = X_train_clean[idx]
        generated_images = generator.predict(noisy_images)
        
        real_labels = np.ones((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1))
        fake_labels = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1))
        
        d_loss_real = discriminator.train_on_batch(clean_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        g_loss = discriminator.train_on_batch(noisy_images, real_labels)
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        
    generator.save("gan_generator.h5")
    discriminator.save("gan_discriminator.h5")
    
    return render(request, "admin/gan_cnn.html", {"cnn_loss": "Trained", "gan_generator_loss": g_losses[-1], "gan_discriminator_loss": d_losses[-1]})



# from django.shortcuts import render
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # Directories
# train_dir = 'dataset/dataset/train'
# val_dir = 'dataset/dataset/test'

# # Parameters
# image_size = (128, 128)
# batch_size = 32
# epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Data transformations
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize(image_size),
#     transforms.RandomRotation(20),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # Load datasets
# train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
# val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # Define LeNet model
# class LeNetmodel(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNetmodel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.pool = nn.AvgPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.fc1 = nn.Linear(16 * 29 * 29, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.dropout = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 16 * 29 * 29)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # Initialize model
# model = LeNetmodel(num_classes=len(train_dataset.classes)).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# def lenet(request):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#     epoch_loss = running_loss / len(train_loader)

#     # Evaluate model
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total

#     print(f"Final Loss: {epoch_loss:.4f}, Final Accuracy: {accuracy:.2f}%")

#     return render(request, 'admin/lenet.html', {'loss': round(epoch_loss, 4), 'accuracy': round(accuracy, 2)})
