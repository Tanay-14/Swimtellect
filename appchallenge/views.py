from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import RegisterForm, UploadForm
from django.contrib.auth.decorators import login_required,user_passes_test
from .swim_analyser import analyze_media, analyze_image_with_variables, generate_zhipu_analysis, generate_video_zhipu_analysis
from .models import UserProfile
from .models import Upload


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
            
            # Create UserProfile with swimming level
            swimming_level = form.cleaned_data["swimming_level"]
            UserProfile.objects.create(user=user, swimming_level=swimming_level)
            
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
            upload = form.save(commit=False)
            upload.username = request.user.username
            upload.save()  # Save the upload instance to get a valid file path
            
            # Get the stroke type from the form
            stroke_type = upload.stroke
            
            # Get the user's swimming level
            try:
                user_profile = UserProfile.objects.get(user=request.user)
                swimming_level = user_profile.swimming_level
            except UserProfile.DoesNotExist:
                swimming_level = 'beginner'  # Default if profile doesn't exist
            
            # Generate AI-powered analysis based on file type and swimming level
            if upload.media.path.lower().endswith((".mp4", ".mov", ".avi")):
                ai_analysis = generate_video_zhipu_analysis(upload.media.path, stroke_type, swimming_level)
            else:
                ai_analysis = generate_zhipu_analysis(upload.media.path, stroke_type, swimming_level)

            upload.analysis_summary = ai_analysis
            upload.save(update_fields=['analysis_summary']) # Update only the analysis field
            
            # Prepare content for template
            content["ai_analysis"] = ai_analysis
            content["stroke_type"] = stroke_type
            content["upload"] = upload
            
    else:
        form = UploadForm()
    content["form"] = form
    return render(request, "upload.html", content)

@login_required
def dashboard(request):
    """Displays the user's upload history."""
    uploads = Upload.objects.filter(username=request.user.username).order_by('-uploaded_at')
    context = {'uploads': uploads}
    return render(request, "dashboard.html", context)

@login_required
def upload_detail(request, upload_id):
    """Displays the full analysis for a single upload, ensuring the upload belongs to the user."""
    upload = get_object_or_404(Upload, pk=upload_id, username=request.user.username)
    context = {'upload': upload}
    return render(request, "upload_detail.html", context)