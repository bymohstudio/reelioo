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

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("This email is already registered. Please login instead.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.set_password(self.cleaned_data['password'])

        base = self.cleaned_data.get('username') or user.email.split('@')[0]
        username = base
        count = 1
        while User.objects.filter(username=username).exists():
            username = f"{base}{count}"
            count += 1
        user.username = username

        if commit:
            user.save()
            profile, created = UserProfile.objects.get_or_create(user=user)
            profile.country = self.cleaned_data['country']
            profile.terms_accepted = True
            profile.save()
        return user


class UserUpdateForm(forms.ModelForm):
    username = forms.CharField(required=True, widget=forms.TextInput(
        attrs={
            'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:border-blue-500 transition shadow-inner placeholder-slate-700'}))

    email = forms.EmailField(required=True, widget=forms.EmailInput(
        attrs={
            'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:border-blue-500 transition shadow-inner placeholder-slate-700'}))

    country = forms.CharField(required=False, widget=forms.TextInput(
        attrs={
            'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:border-blue-500 transition shadow-inner placeholder-slate-700'}))

    password = forms.CharField(required=False, widget=forms.PasswordInput(
        attrs={
            'class': 'w-full bg-[#111] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:border-blue-500 transition shadow-inner placeholder-slate-700',
            'placeholder': 'Leave blank to keep current'}))

    class Meta:
        model = User
        fields = ['username', 'email']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-fill country from profile
        if self.instance and hasattr(self.instance, 'profile'):
            self.fields['country'].initial = self.instance.profile.country

    def save(self, commit=True):
        user = super().save(commit=False)
        # Check if password field is filled
        new_pass = self.cleaned_data.get('password')
        if new_pass:
            user.set_password(new_pass)

        if commit:
            user.save()
            # Save Country
            if hasattr(user, 'profile'):
                user.profile.country = self.cleaned_data.get('country', '')
                user.profile.save()
        return user