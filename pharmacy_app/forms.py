from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Customer, Admin, Medicine, RequestList
from datetime import date

class CustomerRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput, min_length=8)
    confirm_password = forms.CharField(widget=forms.PasswordInput)
    
    class Meta:
        model = Customer
        fields = ['c_fname', 'c_lname', 'c_gender', 'c_birthdate', 'c_phone_no', 'c_email', 'c_address']
        widgets = {
            'c_birthdate': forms.DateInput(attrs={'type': 'date'}),
            'c_phone_no': forms.TextInput(attrs={'pattern': '[0-9]{10}', 'title': 'Please enter 10 digits'}),
        }
    
    def clean_confirm_password(self):
        password = self.cleaned_data.get('password')
        confirm_password = self.cleaned_data.get('confirm_password')
        
        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError("Passwords don't match")
        return confirm_password
    
    def clean_c_phone_no(self):
        phone = self.cleaned_data.get('c_phone_no')
        if len(phone) != 10 or not phone.isdigit():
            raise forms.ValidationError("Phone number must be exactly 10 digits")
        return phone
    
    def clean_c_birthdate(self):
        birthdate = self.cleaned_data.get('c_birthdate')
        if birthdate and birthdate > date.today():
            raise forms.ValidationError("Birth date cannot be in the future")
        return birthdate

class AdminRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput, min_length=8)
    confirm_password = forms.CharField(widget=forms.PasswordInput)
    
    class Meta:
        model = Admin
        fields = ['a_fname', 'a_lname', 'a_gender', 'a_birthdate', 'a_phone_no', 'a_email', 'a_address']
        widgets = {
            'a_birthdate': forms.DateInput(attrs={'type': 'date'}),
            'a_phone_no': forms.TextInput(attrs={'pattern': '[0-9]{10}', 'title': 'Please enter 10 digits'}),
        }
    
    def clean_confirm_password(self):
        password = self.cleaned_data.get('password')
        confirm_password = self.cleaned_data.get('confirm_password')
        
        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError("Passwords don't match")
        return confirm_password
    
    def clean_a_phone_no(self):
        phone = self.cleaned_data.get('a_phone_no')
        if len(phone) != 10 or not phone.isdigit():
            raise forms.ValidationError("Phone number must be exactly 10 digits")
        return phone
    
    def clean_a_birthdate(self):
        birthdate = self.cleaned_data.get('a_birthdate')
        if birthdate and birthdate > date.today():
            raise forms.ValidationError("Birth date cannot be in the future")
        return birthdate

class CustomerLoginForm(forms.Form):
    cust_id = forms.CharField(max_length=20, widget=forms.TextInput(attrs={'placeholder': 'Enter Customer ID'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Enter Password'}))

class AdminLoginForm(forms.Form):
    admin_id = forms.CharField(max_length=20, widget=forms.TextInput(attrs={'placeholder': 'Enter Admin ID'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Enter Password'}))

class MedicineForm(forms.ModelForm):
    class Meta:
        model = Medicine
        fields = ['m_name', 'm_price', 'm_quantity', 'm_mdate', 'm_edate', 'm_descr', 'composition', 'is_generic', 'indications']
        widgets = {
            'm_mdate': forms.DateInput(attrs={'type': 'date'}),
            'm_edate': forms.DateInput(attrs={'type': 'date'}),
            'm_price': forms.NumberInput(attrs={'step': '0.01', 'min': '0'}),
            'm_quantity': forms.NumberInput(attrs={'min': '0'}),
            'composition': forms.Textarea(attrs={'rows': 2}),
            'indications': forms.Textarea(attrs={'rows': 2}),
        }
    
    def clean_m_edate(self):
        mdate = self.cleaned_data.get('m_mdate')
        edate = self.cleaned_data.get('m_edate')
        
        if mdate and edate and edate <= mdate:
            raise forms.ValidationError("Expiry date must be after manufacturing date")
        return edate

class PurchaseForm(forms.Form):
    medicine = forms.ModelChoiceField(queryset=Medicine.objects.all(), empty_label="Select Medicine")
    quantity = forms.IntegerField(min_value=1, widget=forms.NumberInput(attrs={'min': '1'}))

class RequestMedicineForm(forms.ModelForm):
    class Meta:
        model = RequestList
        fields = ['med_name', 'med_quan']
        widgets = {
            'med_quan': forms.NumberInput(attrs={'min': '1'}),
        }

class MLPredictionForm(forms.Form):
    ALGORITHM_CHOICES = [
        ('linear_regression', 'Linear Regression'),
        ('polynomial_regression', 'Polynomial Regression'),
        ('knn', 'K-Nearest Neighbors'),
        ('decision_tree', 'Decision Tree'),
    ]
    
    prediction_type = forms.ChoiceField(
        choices=[('price', 'Price Prediction'), ('quantity', 'Quantity Prediction')],
        widget=forms.RadioSelect
    )
    algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES)
    manufacturing_year = forms.IntegerField(
        min_value=2020, max_value=2030,
        widget=forms.NumberInput(attrs={'min': '2020', 'max': '2030'})
    )
    expiry_year = forms.IntegerField(
        min_value=2021, max_value=2035,
        widget=forms.NumberInput(attrs={'min': '2021', 'max': '2035'})
    )
    price = forms.DecimalField(
        max_digits=10, decimal_places=2, required=False,
        widget=forms.NumberInput(attrs={'step': '0.01', 'min': '0'})
    )
    quantity = forms.IntegerField(
        min_value=1, required=False,
        widget=forms.NumberInput(attrs={'min': '1'})
    )
    
    def clean(self):
        cleaned_data = super().clean()
        prediction_type = cleaned_data.get('prediction_type')
        manufacturing_year = cleaned_data.get('manufacturing_year')
        expiry_year = cleaned_data.get('expiry_year')
        price = cleaned_data.get('price')
        quantity = cleaned_data.get('quantity')
        
        # Check if both years are provided before comparison
        if manufacturing_year is not None and expiry_year is not None:
            if expiry_year <= manufacturing_year:
                raise forms.ValidationError("Expiry year must be after manufacturing year")
        
        if prediction_type == 'price' and not quantity:
            raise forms.ValidationError("Quantity is required for price prediction")
        
        if prediction_type == 'quantity' and not price:
            raise forms.ValidationError("Price is required for quantity prediction")
        
        return cleaned_data


class MedicineSubstitutionForm(forms.Form):
    medicine = forms.ModelChoiceField(queryset=Medicine.objects.all(), empty_label="Select Medicine")
    top_k = forms.IntegerField(min_value=1, max_value=10, initial=5)


class DiseaseRecommendationForm(forms.Form):
    disease_text = forms.CharField(widget=forms.Textarea(attrs={'rows': 3, 'placeholder': 'Enter disease name or symptoms'}))
    top_k = forms.IntegerField(min_value=1, max_value=10, initial=5)

class UserTypeForm(forms.Form):
    USER_TYPE_CHOICES = [
        ('customer', 'Customer'),
        ('admin', 'Admin'),
    ]
    user_type = forms.ChoiceField(
        choices=USER_TYPE_CHOICES,
        widget=forms.RadioSelect,
        initial='customer'
    )
