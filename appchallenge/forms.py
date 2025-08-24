from django import forms
from django.contrib.auth.models import User
from .models import Upload
class RegisterForm(forms.ModelForm):


    password = forms.CharField(widget=forms.PasswordInput)
    password2 = forms.CharField(label="Confirm Password", widget=forms.PasswordInput)


    class Meta:
        model = User
        fields = ["username", "email","password"]

    def clean_password2(self):
        if self.cleaned_data.get("password") != self.cleaned_data.get("password2"):
            raise forms.ValidationError("Passwords do not match")
        return self.cleaned_data.get("password2")



class UploadForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ["media", "stroke"]
