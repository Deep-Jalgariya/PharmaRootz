from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import random
import string

class Customer(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    cust_id = models.CharField(max_length=20, unique=True, primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    c_fname = models.CharField(max_length=50)
    c_lname = models.CharField(max_length=50)
    c_gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    c_birthdate = models.DateField()
    c_phone_no = models.CharField(max_length=15)
    c_email = models.EmailField(unique=True)
    customer_pass = models.CharField(max_length=128)
    c_address = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        if not self.cust_id:
            self.cust_id = f"{self.c_fname}{random.randint(0,9)}{random.randint(0,9)}"
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.cust_id} - {self.c_fname} {self.c_lname}"

class Admin(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    admin_id = models.CharField(max_length=20, unique=True, primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    a_fname = models.CharField(max_length=50)
    a_lname = models.CharField(max_length=50)
    a_gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    a_birthdate = models.DateField()
    a_phone_no = models.CharField(max_length=15)
    a_email = models.EmailField(unique=True)
    admin_pass = models.CharField(max_length=128)
    a_address = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        if not self.admin_id:
            self.admin_id = f"{self.a_fname}{random.randint(0,9)}{random.randint(0,9)}"
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.admin_id} - {self.a_fname} {self.a_lname}"

class Medicine(models.Model):
    m_name = models.CharField(max_length=100, unique=True)
    m_price = models.DecimalField(max_digits=10, decimal_places=2)
    m_quantity = models.IntegerField()
    m_mdate = models.DateField()
    m_edate = models.DateField()
    m_descr = models.TextField()
    composition = models.TextField(blank=True, default="")
    is_generic = models.BooleanField(default=False)
    indications = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.m_name
    
    @property
    def is_expired(self):
        return timezone.now().date() > self.m_edate

class Purchase(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE)
    quantity = models.IntegerField()
    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    purchase_date = models.DateTimeField(auto_now_add=True)
    payment_method = models.CharField(max_length=20, blank=True, null=True)
    payment_reference = models.CharField(max_length=100, blank=True, null=True)
    
    def save(self, *args, **kwargs):
        if not self.total_price:
            self.total_price = self.medicine.m_price * self.quantity
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.customer.cust_id} - {self.medicine.m_name}"

class RequestList(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Completed', 'Completed'),
        ('Rejected', 'Rejected'),
    ]
    
    c_id = models.ForeignKey(Customer, on_delete=models.CASCADE)
    med_name = models.CharField(max_length=100)
    med_quan = models.IntegerField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    request_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.c_id.cust_id} - {self.med_name}"

class MLPrediction(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    medicine_name = models.CharField(max_length=100)
    predicted_price = models.DecimalField(max_digits=10, decimal_places=2)
    predicted_quantity = models.IntegerField()
    algorithm_used = models.CharField(max_length=50)
    prediction_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.customer.cust_id} - {self.medicine_name} - {self.algorithm_used}"

class ModelSearchHistory(models.Model):
    QUERY_TYPE_CHOICES = [
        ("substitution", "Substitution"),
        ("disease", "Disease"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    query_type = models.CharField(max_length=20, choices=QUERY_TYPE_CHOICES)
    input_text = models.TextField(blank=True, default="")
    selected_medicine = models.ForeignKey(Medicine, on_delete=models.SET_NULL, null=True, blank=True)
    results = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        title = f"{self.customer.cust_id} - {self.query_type}"
        if self.selected_medicine:
            title += f" - {self.selected_medicine.m_name}"
        return title

class SubstitutionRecommendation(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    source_medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE, related_name='substitution_source_medicine')
    recommended_medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE, related_name='substitution_recommended_medicine')
    similarity = models.FloatField()
    price_difference = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.customer.cust_id} {self.source_medicine.m_name} → {self.recommended_medicine.m_name} ({self.similarity:.2f})"

class DiseaseRecommendation(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    query_text = models.TextField()
    recommended_medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE)
    similarity = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.customer.cust_id} '{self.query_text[:20]}...' → {self.recommended_medicine.m_name} ({self.similarity:.2f})"
