from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import RegisterForm, UploadForm
from django.contrib.auth.decorators import login_required,user_passes_test
from .swim_analyser import analyze_media, analyze_image_with_variables, generate_zhipu_analysis, generate_video_zhipu_analysis

def landing(request):
    return render(request, "landing.html")

def user_login(request):
    role = request.GET.get("next_role","customer")
    if request.method == "POST":
        username = request.POST.get("username", "")
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("landing")

        else:
            messages.error(request, "Invalid Username or Password")
    return render(request, "login.html")

def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():

            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.save()


            login(request, user)
            return redirect("login")
    else:
        form = RegisterForm()
    return render(request, "register.html", {"form": form})

@login_required
def user_logout(request):
    logout(request)
    return redirect("login")

@login_required
def file_upload(request):
    content = {}
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save()
            
            # Get the stroke type from the form
            stroke_type = upload.stroke
            
            # Generate AI-powered analysis based on file type
            if upload.media.path.lower().endswith((".mp4", ".mov", ".avi")):
                ai_analysis = generate_video_zhipu_analysis(upload.media.path, stroke_type)
            else:
                ai_analysis = generate_zhipu_analysis(upload.media.path, stroke_type)
            
            # Prepare content for template
            content["ai_analysis"] = ai_analysis
            content["stroke_type"] = stroke_type
            content["upload"] = upload
            
    else:
        form = UploadForm()
    content["form"] = form
    return render(request, "upload.html", content)