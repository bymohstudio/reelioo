from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    days_left = serializers.SerializerMethodField()
    access_granted = serializers.SerializerMethodField()

    class Meta:
        model = UserProfile
        fields = ['user', 'is_premium', 'days_left', 'access_granted']

    def get_days_left(self, obj):
        return obj.get_days_left()

    def get_access_granted(self, obj):
        return obj.is_access_granted()