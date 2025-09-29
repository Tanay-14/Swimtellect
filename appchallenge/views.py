from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import RegisterForm, UploadForm
from django.contrib.auth.decorators import login_required,user_passes_test
from .swim_analyser import analyze_media, analyze_image_with_variables, generate_zhipu_analysis, generate_video_zhipu_analysis, generate_training_plan
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
                # For videos, we don't have a single annotated image to show yet.
                # We could enhance this to show an annotated frame. For now, we'll use the original.
                upload.annotated_media_path = upload.media.name
            else:
                ai_analysis = generate_zhipu_analysis(upload.media.path, stroke_type, swimming_level)
                analysis_result = analyze_image_with_variables(upload.media.path)
                if analysis_result.get('annotated_image_path'):
                    upload.annotated_media_path = os.path.relpath(analysis_result['annotated_image_path'], 'media')

            upload.analysis_summary = ai_analysis
            upload.save(update_fields=['analysis_summary', 'annotated_media_path']) # Update analysis and annotated image path
            
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
    user_uploads = Upload.objects.filter(username=request.user.username).order_by('-uploaded_at')
    
    training_plan = request.session.get('training_plan', None)

    context = {
        'uploads': user_uploads,
        'training_plan': training_plan,
    }
    return render(request, "dashboard.html", context)

@login_required
def upload_detail(request, upload_id):
    """Displays the full analysis for a single upload, ensuring the upload belongs to the user."""
    upload = get_object_or_404(Upload, pk=upload_id, username=request.user.username)
    context = {'upload': upload}
    return render(request, "upload_detail.html", context)

@login_required
def training(request):
    """Displays the training page."""
    training_plan = request.session.get('training_plan', None)
    context = {'training_plan': training_plan}
    return render(request, "training.html", context)

@login_required
def generate_plan(request):
    """Generates a new training plan and stores it in the session."""
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        swimming_level = user_profile.swimming_level
    except UserProfile.DoesNotExist:
        swimming_level = 'beginner'

    user_uploads = Upload.objects.filter(username=request.user.username).order_by('-uploaded_at')
    
    plan = generate_training_plan(swimming_level, list(user_uploads))
    request.session['training_plan'] = plan
    return redirect('training')