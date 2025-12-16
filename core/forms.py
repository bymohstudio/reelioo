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

    # --- FIX 1: CASE-INSENSITIVE EMAIL CHECK ---
    def clean_email(self):
        email = self.cleaned_data.get('email')
        # Use __iexact to check for duplicates regardless of Capital/lowercase
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("This email is already registered. Please login instead.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.set_password(self.cleaned_data['password'])

        # --- FIX 2: SMART USERNAME GENERATION (Prevents "Already Exists" Crash) ---
        if self.cleaned_data.get('username'):
            base_username = self.cleaned_data['username']
        else:
            base_username = user.email.split('@')[0]

        # Check if username exists and append number (john -> john1 -> john2)
        username = base_username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1

        user.username = username
        # -------------------------------------------------------------------------

        if commit:
            user.save()
            # Safely create/get profile
            if hasattr(user, 'profile'):
                profile = user.profile
            else:
                profile = UserProfile.objects.create(user=user)

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