from .permissions import is_superadmin, is_admin_or_superadmin

def user_role(request):
    if not request.user.is_authenticated:
        return {'is_superadmin': False, 'can_manage_products': False}
    return {
        'is_superadmin':      is_superadmin(request.user),
        'can_manage_products': is_admin_or_superadmin(request.user),
    }
