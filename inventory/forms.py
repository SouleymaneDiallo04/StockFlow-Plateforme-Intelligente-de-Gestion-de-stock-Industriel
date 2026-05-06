from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User, Group
from .models import Category, Product


class CategoryForm(forms.ModelForm):
    class Meta:
        model  = Category
        fields = ['name', 'description']
        widgets = {
            'name':        forms.TextInput(attrs={'class': 'form-control sf-input', 'placeholder': 'Nom de la catégorie'}),
            'description': forms.Textarea(attrs={'class': 'form-control sf-input', 'rows': 3}),
        }


class ProductForm(forms.ModelForm):
    class Meta:
        model  = Product
        fields = ['name', 'description', 'price', 'stock', 'alert_threshold', 'photo', 'category', 'status', 'ml_sku_id']
        widgets = {
            'name':            forms.TextInput(attrs={'class': 'form-control sf-input'}),
            'description':     forms.Textarea(attrs={'class': 'form-control sf-input', 'rows': 3}),
            'price':           forms.NumberInput(attrs={'class': 'form-control sf-input', 'step': '0.01'}),
            'stock':           forms.NumberInput(attrs={'class': 'form-control sf-input', 'min': '0'}),
            'alert_threshold': forms.NumberInput(attrs={'class': 'form-control sf-input', 'min': '0'}),
            'category':        forms.Select(attrs={'class': 'form-select sf-input'}),
            'status':          forms.Select(attrs={'class': 'form-select sf-input'}),
            'photo':           forms.ClearableFileInput(attrs={'class': 'form-control sf-input', 'id': 'photoInput'}),
            'ml_sku_id':       forms.TextInput(attrs={'class': 'form-control sf-input', 'placeholder': 'ex: COMP-001'}),
        }


class RegisterForm(UserCreationForm):
    ROLE_CHOICES = [
        ('viewer', 'Viewer — consultation uniquement'),
        ('admin',  'Admin — gestion des produits'),
    ]
    role  = forms.ChoiceField(choices=ROLE_CHOICES, widget=forms.Select(attrs={'class': 'form-select sf-input'}))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control sf-input'}))

    class Meta:
        model  = User
        fields = ('username', 'email', 'password1', 'password2', 'role')
        widgets = {'username': forms.TextInput(attrs={'class': 'form-control sf-input'})}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({'class': 'form-control sf-input'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control sf-input'})

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            group, _ = Group.objects.get_or_create(name=self.cleaned_data['role'])
            user.groups.add(group)
        return user
