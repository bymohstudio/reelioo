from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import User
from .services.gumroad_service import verify_gumroad_license


class LicenseKeyBackend(ModelBackend):
    """
    Authenticates using Username + License Key.
    - If user exists: Logs them in if key matches.
    - If user NEW: Verifies key with Gumroad -> Creates Account -> Logs in.
    """

    def authenticate(self, request, username=None, license_key=None, **kwargs):
        if not username or not license_key:
            return None

        # Clean inputs
        username = username.strip()
        license_key = license_key.strip()

        try:
            # --- SCENARIO A: RETURNING USER ---
            user = User.objects.get(username__iexact=username)

            # Check if the provided key matches the one we have on file
            if hasattr(user, 'profile') and user.profile.gumroad_license_key == license_key:
                return user
            else:
                return None  # Wrong key for this user

        except User.DoesNotExist:
            # --- SCENARIO B: NEW USER (SIGN UP) ---
            # 1. Verify the key with Gumroad API first
            is_valid, msg = verify_gumroad_license(license_key)

            if is_valid:
                # 2. Create the user (No password needed)
                # We set a random unusable password so they can't login via standard forms
                user = User.objects.create_user(username=username, email=f"{username}@placeholder.com", password=None)

                # 3. Setup Profile
                user.profile.gumroad_license_key = license_key
                user.profile.is_premium = True
                user.profile.save()
                return user

            return None  # Invalid Key