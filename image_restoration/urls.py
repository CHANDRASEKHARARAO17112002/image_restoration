"""metabolic URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mains import views as m
from users import views as v
from admins import views as a
from django.conf import settings
from django.conf.urls.static import static




urlpatterns = [
    path('admin/', admin.site.urls),
    path('',m.index, name='index'),
    path('about/',m.about, name='about'),
    path('contact/',m.contact, name='contact'),
    path('blog/', m.blog, name='blog'),
    path('blog_details/', m.blog_details, name='blog_details'),
    path('elements/', m.elements, name='elements'),
    path('services/', m.services, name='services'),


    path('register/',v.register, name='register'),
    path('userlogin/',v.userlogin, name='userlogin'),
    path('prediction/',v.prediction,name='prediction'),
    path('udashboard/',v.udashboard,name='udashboard'),


    
    path('adminlogin/',a.adminlogin, name='adminlogin'),
    path('admindashboard/',a.admindashboard, name='admindashboard'),
    path('upload/',a.upload, name='upload'),
    path('gan_cnn/', a.gan_cnn,name='gan_cnn'),
    

]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


