from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from django.utils import timezone
from .models import APIKey

class APIKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        # Get API key from header
        api_key = request.META.get('HTTP_X_API_KEY')
        
        if not api_key:
            return None  # No authentication attempted
        
        try:
            # Find the API key
            key_obj = APIKey.objects.get(key=api_key, is_active=True)
            
            # Update last used timestamp
            key_obj.last_used = timezone.now()
            key_obj.save(update_fields=['last_used'])
            
            # Return user and auth object
            return (key_obj.user, key_obj)
            
        except APIKey.DoesNotExist:
            raise AuthenticationFailed('Invalid API key')