from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import User  # Assuming this is a custom user model; change if using Django's built-in User
from django.core.files.storage import FileSystemStorage

def udashboard(request):
    return render(request, 'user/udashboard.html')

def prediction(req):
    return render(req, 'user/prediction.html')

def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        age = request.POST.get('age')

        profile_picture = request.FILES.get('profile_picture')  # Handle file upload

        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered')
            return redirect('userregister')

        user = User(name=name, email=email, password=password, age=age)

        if profile_picture:
            fs = FileSystemStorage()
            filename = fs.save(profile_picture.name, profile_picture)
            user.profile_picture = filename

        user.save()

        messages.success(request, 'Registration successful! Please login.')
        return redirect('userlogin')

    return render(request, 'user/register.html')



def userlogin(request):
    if request.method == 'POST':
        email = request.POST.get('email')  # Get the username or email
        password = request.POST.get('password')  # Get the password

        # Check if the user exists and the password is correct
        try:
            user = User.objects.get(email=email)
            if user.password == password:  # Be cautious about plain text password comparison
                # Log the user in (you may want to set a session or token here)
                request.session['user_id'] = user.id  # Store user ID in session
                messages.success(request, 'Login successful!')
                return redirect('udashboard')  # Redirect to the index page or desired page
            else:
                messages.error(request, 'Invalid email or password. Please try again.')
        except User.DoesNotExist:
            messages.error(request, 'Invalid email or password. Please try again.')

    return render(request, 'user/userlogin.html')



# # Define your prediction view
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# from django.http import JsonResponse
# from django.core.files.storage import default_storage
# from django.shortcuts import render

# # Define LeNet model
# class LeNet(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.pool = nn.AvgPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.fc1 = nn.Linear(16 * 29 * 29, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 16 * 29 * 29)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Load model
# num_classes = 3
# model = LeNet(num_classes=num_classes)
# model.load_state_dict(torch.load('C:/Users/THANK YOU/Desktop/alzimare/lenet_alzheimer_model.pth'))
# model.eval()

# # Class mapping
# class_mapping = {
#     0: 'AD',
#     1: 'CI',
#     2: 'CN'
# }

# # Image transformation
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# def predict_image(image_file):
#     image_path = default_storage.save('temp/' + image_file.name, image_file)
#     image = Image.open(default_storage.open(image_path))
#     image = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)

#     return class_mapping[predicted.item()]

# def prediction(request):
#     if request.method == 'GET':
#         return render(request, 'user/prediction.html')

#     if request.method == 'POST':
#         image_file = request.FILES.get('image')
#         if not image_file:
#             return JsonResponse({'error': 'No image provided'}, status=400)

#         predicted_class = predict_image(image_file)
#         return JsonResponse({'predicted_class': predicted_class})

# Define your prediction view
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from django.core.files.storage import default_storage
from django.shortcuts import render

# # Define LeNet model
# class LeNet(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.pool = nn.AvgPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.fc1 = nn.Linear(16 * 29 * 29, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 16 * 29 * 29)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Load model
# num_classes = 3
# model = LeNet(num_classes=num_classes)
# model.load_state_dict(torch.load('lenet_alzheimer_model.pth'))
# model.eval()

# # Class mapping
# class_mapping = {
#     0: 'AD',
#     1: 'CI',
#     2: 'CN'
# }

# # Image transformation
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# def predict_image(image_file):
#     image_path = default_storage.save('temp/' + image_file.name, image_file)
#     image = Image.open(default_storage.open(image_path))
#     image = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)

#     return class_mapping[predicted.item()]

def prediction(request):
    return render(request, 'user/prediction.html')
    # if request.method == 'GET':
    #     return render(request, 'user/prediction.html')

    # if request.method == 'POST':
    #     image_file = request.FILES.get('image')
    #     if not image_file:
    #         return render(request, 'user/prediction.html', {'error': 'No image provided'})

    #     predicted_class = predict_image(image_file)
    #     return render(request, 'user/prediction.html', {'predicted_class': predicted_class})




# import os
# import cv2
# import numpy as np
# from django.shortcuts import render
# from django.core.files.storage import default_storage
# from django.conf import settings
# from django.http import JsonResponse
# from tensorflow.keras.models import load_model

# # Load models
# CNN_MODEL_PATH = r"E:/image_restoration/dataset/cnn_denoising_model.h5"
# GAN_MODEL_PATH = r"E:/image_restoration/dataset/gan_generator.h5"

# cnn_model = load_model(CNN_MODEL_PATH)
# gan_model = load_model(GAN_MODEL_PATH)

# IMG_WIDTH, IMG_HEIGHT = 256, 256  # Adjust based on model input size

# def denoise_image(image_path):
#     """Apply CNN and GAN ensemble for denoising."""
#     img = cv2.imread(image_path) / 255.0
#     img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
#     img = np.expand_dims(img, axis=0)  # Add batch dimension

#     # Predict using both models
#     cnn_output = cnn_model.predict(img)[0]
#     gan_output = gan_model.predict(img)[0]

#     # Ensemble: Blend CNN & GAN results (Weighted Average)
#     final_output = (0.6 * cnn_output + 0.4 * gan_output)  # Adjust weights if needed
#     final_output = (final_output * 255).astype(np.uint8)

#     # Save as 'denoised_median.png'
#     output_filename = "denoised_median.png"
#     output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
#     cv2.imwrite(output_path, final_output)

#     return output_filename

# def prediction(request):
#     if request.method == "POST" and request.FILES.get("image"):
#         uploaded_file = request.FILES["image"]
#         file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

#         # Save uploaded image
#         with default_storage.open(file_path, "wb+") as destination:
#             for chunk in uploaded_file.chunks():
#                 destination.write(chunk)

#         # Process image using CNN-GAN ensemble
#         output_image = denoise_image(file_path)

#         return JsonResponse({
#             "uploaded_image": settings.MEDIA_URL + uploaded_file.name,
#             "output_image": settings.MEDIA_URL + output_image
#         })

#     return render(request, "user/prediction.html")denoise nit came


# import os
# import cv2
# import numpy as np
# from django.shortcuts import render
# from django.core.files.storage import default_storage
# from django.conf import settings
# from django.http import JsonResponse
# from tensorflow.keras.models import load_model

# # Load models
# CNN_MODEL_PATH = r"E:/image_restoration/dataset/cnn_denoising_model.h5"
# GAN_MODEL_PATH = r"E:/image_restoration/dataset/gan_generator.h5"

# cnn_model = load_model(CNN_MODEL_PATH)
# gan_model = load_model(GAN_MODEL_PATH)

# IMG_WIDTH, IMG_HEIGHT = 256, 256  # Adjust based on model input size

# def denoise_image(image_path):
#     """Apply CNN and GAN ensemble for denoising."""
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Ensure color mode
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
#     img = img / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Add batch dimension

#     # Predict using both models
#     cnn_output = cnn_model.predict(img)[0]
#     gan_output = gan_model.predict(img)[0]

#     # Ensemble: Blend CNN & GAN results (Weighted Average)
#     final_output = (0.6 * cnn_output + 0.4 * gan_output)  # Adjust weights if needed
#     final_output = (final_output * 255).astype(np.uint8)

#     # Convert back to BGR before saving
#     final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

#     # Save as 'denoised_<original_filename>.png'
#     output_filename = "denoised_" + os.path.basename(image_path)
#     output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
#     cv2.imwrite(output_path, final_output)

#     return output_filename

# def prediction(request):
#     if request.method == "POST" and request.FILES.get("image"):
#         uploaded_file = request.FILES["image"]
#         file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

#         # Save uploaded image
#         with default_storage.open(file_path, "wb+") as destination:
#             for chunk in uploaded_file.chunks():
#                 destination.write(chunk)

#         # Process image using CNN-GAN ensemble
#         output_image = denoise_image(file_path)

#         return JsonResponse({
#             "uploaded_image": settings.MEDIA_URL + uploaded_file.name,
#             "output_image": settings.MEDIA_URL + output_image
#         })

#     return render(request, "user/prediction.html")same denoise not amend

# import os
# import cv2
# import numpy as np
# from django.shortcuts import render
# from django.core.files.storage import default_storage
# from django.core.files.base import ContentFile
# from tensorflow.keras.models import load_model

# # Load models
# cnn_model_path = r"E:/image_restoration/dataset/cnn_denoising_model.h5"
# gan_generator_path = r"E:/image_restoration/dataset/gan_generator.h5"
# generator = load_model(gan_generator_path)

# IMG_WIDTH, IMG_HEIGHT = 256, 256  # Adjust as per model input size

# def test_denoising(image_path):
#     img = cv2.imread(image_path) / 255.0
#     img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
#     img = np.expand_dims(img, axis=0)  # Add batch dimension

#     denoised_img = generator.predict(img)[0]  # Remove batch dim
#     denoised_img = (denoised_img * 255).astype(np.uint8)

#     output_path = os.path.join("media", "denoised_output.png")
#     cv2.imwrite(output_path, denoised_img)
#     return output_path

# def remove_salt_pepper_noise(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     denoised_img = cv2.medianBlur(img, 3)  # Apply median filter
#     output_path = os.path.join("media", "denoised_median.png")
#     cv2.imwrite(output_path, denoised_img)
#     return output_path

# def prediction(request):
#     if request.method == 'POST' and request.FILES['image']:
#         image_file = request.FILES['image']
#         file_name = default_storage.save(f"uploads/{image_file.name}", ContentFile(image_file.read()))
#         image_path = default_storage.path(file_name)
        
#         # Apply denoising methods
#         denoised_image_path = test_denoising(image_path)
#         filtered_image_path = remove_salt_pepper_noise(image_path)
        
#         return render(request, 'user/prediction.html', {
#             'original_image': file_name,
#             'denoised_image': denoised_image_path,
#             'filtered_image': filtered_image_path,
#         })
    
#     return render(request, 'user/prediction.html') any image not came



# import os
# import cv2
# import numpy as np
# from django.shortcuts import render
# from django.core.files.storage import default_storage
# from django.conf import settings
# from django.http import JsonResponse
# from tensorflow.keras.models import load_model

# def prediction(request):
#     if request.method == "POST" and request.FILES.get("image"):
#         # Load models
#         CNN_MODEL_PATH = r"E:/image_restoration/dataset/cnn_denoising_model.h5"
#         GAN_MODEL_PATH = r"E:/image_restoration/dataset/gan_generator.h5"
        
#         try:
#             cnn_model = load_model(CNN_MODEL_PATH)
#             gan_model = load_model(GAN_MODEL_PATH)
#             print("✅ Models Loaded Successfully!")
#         except Exception as e:
#             print("❌ Error loading models:", e)
#             return JsonResponse({"error": "Model loading failed"}, status=500)

#         IMG_WIDTH, IMG_HEIGHT = 256, 256  # Adjust based on model input size
        
#         def remove_salt_pepper_noise(image_path):
#             """Apply median filter to remove salt & pepper noise."""
#             try:
#                 img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#                 if img is None:
#                     print(f"❌ Error: Could not read image from {image_path}")
#                     return None

#                 denoised_img = cv2.medianBlur(img, 3)  # Apply median filter
#                 filtered_filename = "filtered_" + os.path.basename(image_path)
#                 filtered_path = os.path.join(settings.MEDIA_ROOT, filtered_filename)
#                 cv2.imwrite(filtered_path, denoised_img)
#                 return filtered_filename
#             except Exception as e:
#                 print("❌ Error in salt-pepper removal:", e)
#                 return None
        
#         def denoise_image(image_path):
#             """Apply CNN and GAN ensemble for denoising."""
#             try:
#                 img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#                 if img is None:
#                     print(f"❌ Error: Could not read image from {image_path}")
#                     return None
                
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#                 img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
#                 img = img / 255.0  # Normalize
#                 img = np.expand_dims(img, axis=0)  # Add batch dimension

#                 # Predict using both models
#                 cnn_output = cnn_model.predict(img)[0]
#                 gan_output = gan_model.predict(img)[0]

#                 # Ensemble: Blend CNN & GAN results (Weighted Average)
#                 final_output = (0.6 * cnn_output + 0.4 * gan_output)  # Adjust weights if needed
#                 final_output = (final_output * 255).astype(np.uint8)

#                 # Convert back to BGR before saving
#                 final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

#                 # Save as 'denoised_*.png'
#                 output_filename = "denoised_" + os.path.basename(image_path)
#                 output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
#                 cv2.imwrite(output_path, final_output)

#                 return output_filename
#             except Exception as e:
#                 print("❌ Error in denoising:", e)
#                 return None
        
#         uploaded_file = request.FILES["image"]
#         file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

#         # Save uploaded image
#         with default_storage.open(file_path, "wb+") as destination:
#             for chunk in uploaded_file.chunks():
#                 destination.write(chunk)

#         # Step 1: Remove salt & pepper noise
#         filtered_image = remove_salt_pepper_noise(file_path)
#         if not filtered_image:
#             return JsonResponse({"error": "Salt & Pepper noise removal failed"}, status=500)

#         filtered_image_path = os.path.join(settings.MEDIA_ROOT, filtered_image)

#         # Step 2: Apply CNN-GAN denoising
#         output_image = denoise_image(filtered_image_path)
#         if not output_image:
#             return JsonResponse({"error": "Denoising process failed"}, status=500)

#         return JsonResponse({
#             "uploaded_image": settings.MEDIA_URL + uploaded_file.name,
#             "filtered_image": settings.MEDIA_URL + filtered_image,
#             "output_image": settings.MEDIA_URL + output_image
#         })

#     return render(request, "user/prediction.html")



# import os
# import cv2
# import numpy as np
# from django.shortcuts import render
# from django.core.files.storage import default_storage
# from django.conf import settings
# from tensorflow.keras.models import load_model
# from django.http import JsonResponse

# # Load models
# CNN_MODEL_PATH = r"E:/image_restoration/dataset/cnn_denoising_model.h5"
# GAN_MODEL_PATH = r"E:/image_restoration/dataset/gan_generator.h5"

# try:
#     cnn_model = load_model(CNN_MODEL_PATH)
#     gan_model = load_model(GAN_MODEL_PATH)
#     print("✅ Models Loaded Successfully!")
# except Exception as e:
#     print("❌ Error loading models:", e)

# IMG_WIDTH, IMG_HEIGHT = 256, 256  # Adjust based on model input size

# def remove_salt_pepper_noise(image_path):
#     """Apply median filter to remove salt & pepper noise."""
#     try:
#         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         if img is None:
#             print(f"❌ Error: Could not read image from {image_path}")
#             return None

#         denoised_img = cv2.medianBlur(img, 3)  # Apply median filter
#         filtered_filename = "filtered_" + os.path.basename(image_path)
#         filtered_path = os.path.join(settings.MEDIA_ROOT, filtered_filename)
        
#         # Ensure media directory exists
#         os.makedirs(os.path.dirname(filtered_path), exist_ok=True)

#         cv2.imwrite(filtered_path, denoised_img)
#         return filtered_filename
#     except Exception as e:
#         print("❌ Error in salt-pepper removal:", e)
#         return None

# def denoise_image(image_path):
#     """Apply CNN and GAN ensemble for denoising."""
#     try:
#         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         if img is None:
#             print(f"❌ Error: Could not read image from {image_path}")
#             return None

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#         img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
#         img = img.astype(np.float32) / 255.0  # Normalize
#         img = np.expand_dims(img, axis=0)  # Add batch dimension

#         # Predict using both models
#         cnn_output = cnn_model.predict(img)[0]
#         gan_output = gan_model.predict(img)[0]

#         # Ensemble: Blend CNN & GAN results (Weighted Average)
#         final_output = (0.6 * cnn_output + 0.4 * gan_output)
#         final_output = (final_output * 255).astype(np.uint8)

#         # Convert back to BGR before saving
#         final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

#         # Save as 'denoised_*.png'
#         output_filename = "denoised_" + os.path.basename(image_path)
#         output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

#         # Ensure media directory exists
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         cv2.imwrite(output_path, final_output)

#         return output_filename
#     except Exception as e:
#         print("❌ Error in denoising:", e)
#         return None

# def prediction(request):
#     if request.method == "POST" and request.FILES.get("image"):
#         uploaded_file = request.FILES["image"]
        
#         # Save uploaded image using Django's default storage
#         file_name = default_storage.save(f"uploads/{uploaded_file.name}", uploaded_file)
#         file_path = default_storage.path(file_name)

#         # Step 1: Remove salt & pepper noise
#         filtered_image = remove_salt_pepper_noise(file_path)
#         if not filtered_image:
#             return JsonResponse({"error": "Salt & Pepper noise removal failed"}, status=500)

#         filtered_image_path = os.path.join(settings.MEDIA_ROOT, filtered_image)

#         # Step 2: Apply CNN-GAN denoising
#         output_image = denoise_image(filtered_image_path)
#         if not output_image:
#             return JsonResponse({"error": "Denoising process failed"}, status=500)

#         return JsonResponse({
#             "uploaded_image": settings.MEDIA_URL + file_name,
#             "filtered_image": settings.MEDIA_URL + filtered_image,
#             "output_image": settings.MEDIA_URL + output_image
#         })

#     return render(request, "user/prediction.html") worked  but json

import os
import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from tensorflow.keras.models import load_model

# Load models
CNN_MODEL_PATH = r"E:/image_restoration/dataset/cnn_denoising_model.h5"
GAN_MODEL_PATH = r"E:/image_restoration/dataset/gan_generator.h5"

try:
    cnn_model = load_model(CNN_MODEL_PATH)
    gan_model = load_model(GAN_MODEL_PATH)
    print("✅ Models Loaded Successfully!")
except Exception as e:
    print("❌ Error loading models:", e)

IMG_WIDTH, IMG_HEIGHT = 256, 256  # Adjust based on model input size

def remove_salt_pepper_noise(image_path):
    """Apply median filter to remove salt & pepper noise."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Error: Could not read image from {image_path}")
            return None

        denoised_img = cv2.medianBlur(img, 3)  # Apply median filter
        filtered_filename = "filtered_" + os.path.basename(image_path)
        filtered_path = os.path.join(settings.MEDIA_ROOT, filtered_filename)
        
        # Ensure media directory exists
        os.makedirs(os.path.dirname(filtered_path), exist_ok=True)

        cv2.imwrite(filtered_path, denoised_img)
        return filtered_filename
    except Exception as e:
        print("❌ Error in salt-pepper removal:", e)
        return None

def denoise_image(image_path):
    """Apply CNN and GAN ensemble for denoising."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Error: Could not read image from {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict using both models
        cnn_output = cnn_model.predict(img)[0]
        gan_output = gan_model.predict(img)[0]

        # Ensemble: Blend CNN & GAN results (Weighted Average)
        final_output = (0.6 * cnn_output + 0.4 * gan_output)
        final_output = (final_output * 255).astype(np.uint8)

        # Convert back to BGR before saving
        final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

        # Save as 'denoised_*.png'
        output_filename = "denoised_" + os.path.basename(image_path)
        output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

        # Ensure media directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, final_output)

        return output_filename
    except Exception as e:
        print("❌ Error in denoising:", e)
        return None

def prediction(request):
    if request.method == "POST" and request.FILES.get("image"):
        uploaded_file = request.FILES["image"]
        
        # Save uploaded image using Django's default storage
        file_name = default_storage.save(f"uploads/{uploaded_file.name}", uploaded_file)
        file_path = default_storage.path(file_name)

        # Step 1: Remove salt & pepper noise
        filtered_image = remove_salt_pepper_noise(file_path)
        if not filtered_image:
            return render(request, "user/prediction.html", {"error": "Salt & Pepper noise removal failed"})

        filtered_image_path = os.path.join(settings.MEDIA_ROOT, filtered_image)

        # Step 2: Apply CNN-GAN denoising
        output_image = denoise_image(filtered_image_path)
        if not output_image:
            return render(request, "user/prediction.html", {"error": "Denoising process failed"})

        return render(request, "user/prediction.html", {
            "uploaded_image": settings.MEDIA_URL + file_name,
            "filtered_image": settings.MEDIA_URL + filtered_image,
            "output_image": settings.MEDIA_URL + output_image
        })

    return render(request, "user/prediction.html")

