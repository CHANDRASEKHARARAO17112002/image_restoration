from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.shortcuts import render,redirect
import urllib.request
import urllib.parse
import random 
import ssl
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings

from django.shortcuts import render

def index(request):
    return render(request, 'main/index.html')  # Adjust the path if your HTML file is named differently

def about(request):
    return render(request, 'main/about.html')  # Create 'about.html' as needed

def services(request):
    return render(request, 'main/services.html')  # Create 'services.html' as needed

def blog(request):
    return render(request, 'main/blog.html')  # Create 'blog.html' as needed

def contact(request):
    return render(request, 'main/contact.html')  # Create 'contact.html' as needed

def blog_details(request):
    return render(request, 'main/blog_details.html')  # Create 'blog_details.html' as needed

def elements(request):
    return render(request, 'main/elements.html') 