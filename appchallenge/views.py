import os
import json
import re
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import RegisterForm, UploadForm
from django.contrib.auth.decorators import login_required,user_passes_test
from .swim_analyser import analyze_media, analyze_image_with_variables, generate_zhipu_analysis, generate_video_zhipu_analysis, generate_training_plan
from .models import UserProfile
from .models import Upload, UserProfile
from .youtube import get_youtube_videos


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

def _update_user_scores(user_profile, new_scores):
    """
    Updates the user's scores in their profile.
    """
    score_mapping = {
        "Coordination": "coordination_score",
        "Breathing technique": "breathing_technique_score",
        "Body alignment": "body_alignment_score",
        "Arm stroke efficiency": "arm_stroke_efficiency_score",
        "Kick technique": "kick_technique_score",
    }
    for score_name, score_value in new_scores.items():
        if model_field_name := score_mapping.get(score_name):
            setattr(user_profile, model_field_name, score_value)
    user_profile.save()
@login_required
def file_upload(request):
    content = {}
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save(commit=False)
            upload.username = request.user.username
            upload.save()  
            
            #Get the type of stroke from the form
            stroke_type = upload.stroke
            
            # Get the user's swimming level
            try:
                user_profile = UserProfile.objects.get(user=request.user)
                swimming_level = user_profile.swimming_level
            except UserProfile.DoesNotExist:
                swimming_level = 'beginner'  # Default
            
            #Analyze
            if upload.media.path.lower().endswith((".mp4", ".mov", ".avi")):
                ai_analysis = generate_video_zhipu_analysis(upload.media.path, stroke_type, swimming_level)
                upload.annotated_media_path = upload.media.name
            else:
                ai_analysis = generate_zhipu_analysis(upload.media.path, stroke_type, swimming_level)
                analysis_result = analyze_image_with_variables(upload.media.path)
                if analysis_result.get('annotated_image_path'):
                    upload.annotated_media_path = os.path.relpath(analysis_result['annotated_image_path'], 'media')

            # Extract scores from AI analysis and update user profile
            try:
                
                score_match = re.search(r"```json\s*(\{.*?\})\s*```", ai_analysis, re.DOTALL)
                if score_match:
                    scores_json = json.loads(score_match.group(1))
                    new_scores = scores_json.get("scores")
                    if new_scores:
                        upload.scores = new_scores  # Save scores to the Upload model
                        _update_user_scores(user_profile, new_scores)
                        
                        ai_analysis = ai_analysis.replace(score_match.group(0), "").strip()

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Could not parse scores from AI response: {e}")
            

            upload.analysis_summary = ai_analysis
            upload.save(update_fields=['analysis_summary', 'annotated_media_path', 'scores']) # Update analysis and annotated image path
            
            content["ai_analysis"] = ai_analysis
            content["stroke_type"] = stroke_type
            content["upload"] = upload
            
    else:
        form = UploadForm()
    content["form"] = form
    return render(request, "upload.html", content)

@login_required
def dashboard(request):
    
    user_uploads = Upload.objects.filter(username=request.user.username).order_by('-uploaded_at')
    
    training_plan = request.session.get('training_plan', None)

    context = {
        'uploads': user_uploads,
        'training_plan': training_plan,
    }
    return render(request, "dashboard.html", context)

@login_required
def upload_detail(request, upload_id):
    upload = get_object_or_404(Upload, pk=upload_id, username=request.user.username)
    context = {'upload': upload}
    return render(request, "upload_detail.html", context)

@login_required
def training(request):
    training_plan = request.session.get('training_plan', None)
    user_uploads = Upload.objects.filter(username=request.user.username, scores__isnull=False)

    score_labels = [
        "Coordination", "Breathing technique", "Body alignment",
        "Arm stroke efficiency", "Kick technique"
    ]
    
    if user_uploads.exists():
        total_scores = {label: [] for label in score_labels}
        for upload in user_uploads:
            if upload.scores:
                for category, score in upload.scores.items():
                    if category in total_scores:
                        total_scores[category].append(score)

        avg_scores = {}
        for category, scores in total_scores.items():
            if scores:
                avg_scores[category] = sum(scores) / len(scores)
            else:
                avg_scores[category] = 0

        score_values = [avg_scores.get(label, 0) for label in score_labels]
    else:
        # Fallback to user profile scores if no uploads have scores
        user_profile = get_object_or_404(UserProfile, user=request.user)
        score_values = [
            user_profile.coordination_score,
            user_profile.breathing_technique_score,
            user_profile.body_alignment_score,
            user_profile.arm_stroke_efficiency_score,
            user_profile.kick_technique_score,
        ]

    scores_with_labels = sorted(zip(score_labels, score_values), key=lambda item: item[1])
    areas_for_improvement = scores_with_labels[:2]

  
    videos = []
    
    if areas_for_improvement:
        
        improvement_area_name = areas_for_improvement[0][0]
        query = f"swimming {improvement_area_name} drill"
        try:
            
            videos = get_youtube_videos(query, max_results=3)
        except Exception as e:
            print(f"Error fetching YouTube videos: {e}")
            

    context = {
        'training_plan': training_plan,
        'score_labels': json.dumps(score_labels),
        'score_values': json.dumps(score_values),
        'areas_for_improvement': areas_for_improvement,
        'videos': videos,
    }
    return render(request, "training.html", context)

@login_required
def generate_plan(request):
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        swimming_level = user_profile.swimming_level
    except UserProfile.DoesNotExist:
        swimming_level = 'beginner'

    user_uploads = Upload.objects.filter(username=request.user.username).order_by('-uploaded_at')
    
    plan = generate_training_plan(swimming_level, list(user_uploads))
    request.session['training_plan'] = plan
    return redirect('training')