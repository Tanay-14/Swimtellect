from django.db import models
from django.contrib.auth.models import User
#kjsdhfksdjhfjk
class UserProfile(models.Model):
    SWIMMING_LEVEL_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
        ('competitive', 'Competitive'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)

    swimming_level = models.CharField(max_length=20, choices=SWIMMING_LEVEL_CHOICES, default='beginner')
    
    def __str__(self):
        return f"{self.user.username} - {self.get_swimming_level_display()}"

class Upload(models.Model):
    STROKE_CHOICES = [
        ('freestyle', 'Freestyle'),
        ('backstroke', 'Backstroke'),
        ('breaststroke', 'Breaststroke'),
        ('butterfly', 'Butterfly'),
    ]
    #user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    username = models.CharField(max_length=150, null=True, blank=True)
 
    media = models.FileField(upload_to="uploads/")
    stroke = models.CharField(max_length=20, choices=STROKE_CHOICES, default='freestyle')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analysis_summary = models.TextField(blank=True, null=True)
    annotated_media_path = models.CharField(max_length=255, blank=True, null=True)