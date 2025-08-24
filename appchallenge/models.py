from django.db import models

class Upload(models.Model):
    STROKE_CHOICES = [
        ('freestyle', 'Freestyle'),
        ('backstroke', 'Backstroke'),
        ('breaststroke', 'Breaststroke'),
        ('butterfly', 'Butterfly'),
    ]
    
    media = models.FileField(upload_to="uploads/")
    stroke = models.CharField(max_length=20, choices=STROKE_CHOICES, default='freestyle')
    uploaded_at = models.DateTimeField(auto_now_add=True)