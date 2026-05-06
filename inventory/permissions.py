def is_superadmin(user):
    return user.is_authenticated and (
        user.is_superuser or user.groups.filter(name='superadmin').exists()
    )

def is_admin_or_superadmin(user):
    return user.is_authenticated and (
        user.is_superuser or user.groups.filter(name__in=['superadmin', 'admin']).exists()
    )
