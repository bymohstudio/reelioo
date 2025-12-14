from django import forms
from django.contrib.auth.models import User
from core.models import UserProfile


class SignupForm(forms.ModelForm):
    username = forms.CharField(required=False, widget=forms.TextInput(attrs={'placeholder': 'Optional Username',
                                                                             'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:border-blue-500 transition'}))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'placeholder': 'analyst@fund.com',
                                                                           'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:border-blue-500 transition'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': '••••••••',
                                                                 'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:border-blue-500 transition'}))
    country = forms.CharField(required=True, widget=forms.TextInput(attrs={'placeholder': 'Country (e.g. India)',
                                                                           'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:border-blue-500 transition'}))
    terms_agreement = forms.BooleanField(required=True, error_messages={
        'required': 'You must accept the Terms of Service to continue.'})

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def save(self, commit=True):
        # Handle User Creation
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        # If no username provided, use email prefix or email itself
        if not self.cleaned_data['username']:
            user.username = user.email.split('@')[0]
        else:
            user.username = self.cleaned_data['username']

        user.set_password(self.cleaned_data['password'])

        if commit:
            user.save()
            # Update Profile
            profile = user.profile
            profile.country = self.cleaned_data['country']
            profile.terms_accepted = True
            profile.save()

        return user


class UserUpdateForm(forms.ModelForm):
    username = forms.CharField(required=True, widget=forms.TextInput(
        attrs={'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm'}))
    email = forms.EmailField(required=True, widget=forms.EmailInput(
        attrs={'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm'}))
    country = forms.CharField(required=False, widget=forms.TextInput(
        attrs={'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm'}))

    class Meta:
        model = User
        fields = ['username', 'email']