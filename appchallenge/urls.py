"""
URL configuration for appchallenge project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path("login/", views.user_login, name ="login"),
    path("register/", views.register, name="register"),
    path('logout/', views.user_logout, name="logout"),
    path("upload/", views.file_upload, name="upload"),
    path("upload/<int:upload_id>/", views.upload_detail, name="upload_detail"),
    path("training/", views.training, name="training"),
    path("generate-plan/", views.generate_plan, name="generate_plan"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("contact/", views.contact, name="contact"),
    path("", views.landing, name="landing"),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)