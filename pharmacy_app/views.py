from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.db.models import Sum
from django.utils import timezone
from datetime import datetime, date
import json
from .models import Customer, Admin, Medicine, Purchase, RequestList, MLPrediction, ModelSearchHistory, SubstitutionRecommendation, DiseaseRecommendation
from .forms import (
    CustomerRegistrationForm, AdminRegistrationForm, CustomerLoginForm, 
    AdminLoginForm, MedicineForm, PurchaseForm, RequestMedicineForm, 
    MLPredictionForm, UserTypeForm, MedicineSubstitutionForm, DiseaseRecommendationForm
)
from .utils import ml_predictor, substitution_recommender, disease_recommender
import hashlib
from functools import wraps
from decimal import Decimal
from django.views.decorators.csrf import csrf_exempt
from random import randint

def login_required(view_func):
    """Custom decorator to check if user is logged in"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if 'customer_id' in request.session or 'admin_id' in request.session:
            return view_func(request, *args, **kwargs)
        else:
            messages.error(request, 'Please login to access this page.')
            return redirect('home')
    return wrapper

def home(request):
    """Home page view"""
    medicines = Medicine.objects.filter(m_quantity__gt=0)[:6]  # Show 6 available medicines
    context = {
        'medicines': medicines,
        'total_medicines': Medicine.objects.count(),
        'total_customers': Customer.objects.count(),
    }
    return render(request, 'pharmacy_app/home.html', context)

def user_type_selection(request):
    """View for user to select whether they are customer or admin"""
    if request.method == 'POST':
        form = UserTypeForm(request.POST)
        if form.is_valid():
            user_type = form.cleaned_data['user_type']
            if user_type == 'customer':
                return redirect('customer_register')
            else:
                return redirect('admin_register')
    else:
        form = UserTypeForm()
    
    return render(request, 'pharmacy_app/user_type_selection.html', {'form': form})

def customer_register(request):
    """Customer registration view"""
    if request.method == 'POST':
        form = CustomerRegistrationForm(request.POST)
        if form.is_valid():
            customer = form.save(commit=False)
            
            # Hash password
            password = form.cleaned_data['password']
            customer.customer_pass = hashlib.sha256(password.encode()).hexdigest()
            
            # Generate customer ID
            customer.save()
            
            messages.success(request, f'Registration successful! Your Customer ID is: {customer.cust_id}')
            return redirect('customer_login')
    else:
        form = CustomerRegistrationForm()
    
    return render(request, 'pharmacy_app/customer_register.html', {'form': form})

def admin_register(request):
    """Admin registration view"""
    if Admin.objects.count() >= 2:
        messages.error(request, 'Admin registration limit reached. Only 2 admins are allowed.')
        return redirect('admin_login')
    if request.method == 'POST':
        form = AdminRegistrationForm(request.POST)
        if form.is_valid():
            admin = form.save(commit=False)
            
            # Hash password
            password = form.cleaned_data['password']
            admin.admin_pass = hashlib.sha256(password.encode()).hexdigest()
            
            # Generate admin ID
            admin.save()
            
            messages.success(request, f'Registration successful! Your Admin ID is: {admin.admin_id}')
            return redirect('admin_login')
    else:
        form = AdminRegistrationForm()
    
    return render(request, 'pharmacy_app/admin_register.html', {'form': form, 'user_type': request.session.get('user_type')})

def customer_login(request):
    """Customer login view"""
    if request.method == 'POST':
        form = CustomerLoginForm(request.POST)
        if form.is_valid():
            cust_id = form.cleaned_data['cust_id']
            password = form.cleaned_data['password']
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            try:
                customer = Customer.objects.get(cust_id=cust_id, customer_pass=hashed_password)
                request.session['customer_id'] = customer.cust_id
                request.session['user_type'] = 'customer'
                messages.success(request, f'Welcome back, {customer.c_fname}!')
                return redirect('customer_dashboard')
            except Customer.DoesNotExist:
                messages.error(request, 'Invalid Customer ID or Password')
    else:
        form = CustomerLoginForm()
    
    return render(request, 'pharmacy_app/customer_login.html', {'form': form})

def admin_login(request):
    """Admin login view"""
    if request.method == 'POST':
        form = AdminLoginForm(request.POST)
        if form.is_valid():
            admin_id = form.cleaned_data['admin_id']
            password = form.cleaned_data['password']
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            try:
                admin = Admin.objects.get(admin_id=admin_id, admin_pass=hashed_password)
                request.session['admin_id'] = admin.admin_id
                request.session['user_type'] = 'admin'
                messages.success(request, f'Welcome back, {admin.a_fname}!')
                return redirect('admin_dashboard')
            except Admin.DoesNotExist:
                messages.error(request, 'Invalid Admin ID or Password')
    else:
        form = AdminLoginForm()
    
    return render(request, 'pharmacy_app/admin_login.html', {'form': form})

def logout_view(request):
    """Logout view"""
    logout(request)
    request.session.flush()
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')

@login_required
def customer_dashboard(request):
    """Customer dashboard view"""
    if 'customer_id' not in request.session or request.session.get('user_type') != 'customer':
        messages.error(request, 'Please login as customer to access this page.')
        return redirect('customer_login')
    
    customer = get_object_or_404(Customer, cust_id=request.session['customer_id'])
    purchases = Purchase.objects.filter(customer=customer).order_by('-purchase_date')
    total_spent = purchases.aggregate(Sum('total_price'))['total_price__sum'] or 0
    recent_requests = RequestList.objects.filter(c_id=customer).order_by('-request_date')[:5]
    recent_medicines = Medicine.objects.filter(m_quantity__gt=0)
    recent_ai_subs = SubstitutionRecommendation.objects.filter(customer=customer).order_by('-created_at')[:5]
    recent_ai_disease = DiseaseRecommendation.objects.filter(customer=customer).order_by('-created_at')[:5]
    all_medicines = Medicine.objects.all().order_by('m_name')
    
    context = {
        'customer': customer,
        'purchases': purchases[:5],
        'total_spent': total_spent,
        'recent_medicines': recent_medicines,
        'recent_requests': recent_requests,
        'all_medicines': all_medicines,
        'user_type': request.session.get('user_type'),
        'recent_ai_subs': recent_ai_subs,
        'recent_ai_disease': recent_ai_disease,
    }
    return render(request, 'pharmacy_app/customer_dashboard.html', context)

@login_required
def admin_dashboard(request):
    """Admin dashboard view"""
    if 'admin_id' not in request.session or request.session.get('user_type') != 'admin':
        messages.error(request, 'Please login as admin to access this page.')
        return redirect('admin_login')
    
    admin = get_object_or_404(Admin, admin_id=request.session['admin_id'])
    total_medicines = Medicine.objects.count()
    low_stock_medicines = Medicine.objects.filter(m_quantity__lt=10)
    expired_medicines = Medicine.objects.filter(m_edate__lt=date.today())
    pending_requests = RequestList.objects.filter(status='Pending').count()
    recent_pending_requests = RequestList.objects.filter(status='Pending').order_by('-request_date')
    
    context = {
        'admin': admin,
        'total_medicines': total_medicines,
        'low_stock_medicines': low_stock_medicines,
        'expired_medicines': expired_medicines,
        'pending_requests': pending_requests,
        'recent_pending_requests': recent_pending_requests,
        'user_type': request.session.get('user_type'),
    }
    return render(request, 'pharmacy_app/admin_dashboard.html', context)

@login_required
def buy_medicine(request):
    """Buy medicine view for customers"""
    if 'customer_id' not in request.session or request.session.get('user_type') != 'customer':
        messages.error(request, 'Please login as customer to access this page.')
        return redirect('customer_login')
    
    if request.method == 'POST':
        form = PurchaseForm(request.POST)
        if form.is_valid():
            medicine = form.cleaned_data['medicine']
            quantity = form.cleaned_data['quantity']
            if quantity > medicine.m_quantity:
                messages.error(request, f'Only {medicine.m_quantity} units available for {medicine.m_name}')
            else:
                # Store pending purchase in session and redirect to payment page
                request.session['pending_medicine_id'] = medicine.id
                request.session['pending_quantity'] = quantity
                return redirect('payment_page')
    else:
        form = PurchaseForm()
    
    medicines = Medicine.objects.filter(m_quantity__gt=0)
    return render(request, 'pharmacy_app/buy_medicine.html', {'form': form, 'medicines': medicines, 'user_type': request.session.get('user_type')})

@login_required
def show_medicines(request):
    """Show all available medicines"""
    medicines = Medicine.objects.filter(m_quantity__gt=0)
    return render(request, 'pharmacy_app/show_medicines.html', {'medicines': medicines, 'user_type': request.session.get('user_type')})

@login_required
def request_medicine(request):
    """Request medicine view for customers"""
    if 'customer_id' not in request.session or request.session.get('user_type') != 'customer':
        messages.error(request, 'Please login as customer to access this page.')
        return redirect('customer_login')
    
    if request.method == 'POST':
        form = RequestMedicineForm(request.POST)
        if form.is_valid():
            request_med = form.save(commit=False)
            request_med.c_id = get_object_or_404(Customer, cust_id=request.session['customer_id'])
            request_med.save()
            
            messages.success(request, 'Medicine request submitted successfully!')
            return redirect('customer_dashboard')
    else:
        form = RequestMedicineForm()
    
    # Get user's previous requests
    customer = get_object_or_404(Customer, cust_id=request.session['customer_id'])
    user_requests = RequestList.objects.filter(c_id=customer).order_by('-request_date')
    
    return render(request, 'pharmacy_app/request_medicine.html', {
        'form': form,
        'user_requests': user_requests,
        'user_type': request.session.get('user_type'),
    })

@login_required
def ml_prediction(request):
    """Machine learning prediction view for customers"""
    if 'customer_id' not in request.session or request.session.get('user_type') != 'customer':
        messages.error(request, 'Please login as customer to access this page.')
        return redirect('customer_login')
    
    prediction_result = None
    charts = {}
    
    if request.method == 'POST':
        form = MLPredictionForm(request.POST)
        if form.is_valid():
            prediction_type = form.cleaned_data['prediction_type']
            algorithm = form.cleaned_data['algorithm']
            manufacturing_year = form.cleaned_data['manufacturing_year']
            expiry_year = form.cleaned_data['expiry_year']
            
            try:
                if prediction_type == 'price':
                    quantity = form.cleaned_data['quantity']
                    predicted_price = ml_predictor.predict_price(
                        manufacturing_year, expiry_year, quantity, algorithm
                    )
                    prediction_result = {
                        'type': 'Price',
                        'value': f'₹{predicted_price:.2f}',
                        'algorithm': algorithm.replace('_', ' ').title(),
                        'details': f'For {quantity} units, manufactured in {manufacturing_year}, expiring in {expiry_year}'
                    }
                else:  # quantity prediction
                    price = form.cleaned_data['price']
                    predicted_quantity = ml_predictor.predict_quantity(
                        manufacturing_year, expiry_year, price, algorithm
                    )
                    prediction_result = {
                        'type': 'Quantity',
                        'value': predicted_quantity,
                        'algorithm': algorithm.replace('_', ' ').title(),
                        'details': f'At price ₹{price}, manufactured in {manufacturing_year}, expiring in {expiry_year}'
                    }
                
                # Save prediction to database
                customer = get_object_or_404(Customer, cust_id=request.session['customer_id'])
                MLPrediction.objects.create(
                    customer=customer,
                    medicine_name=f"Predicted {prediction_type}",
                    predicted_price=predicted_price if prediction_type == 'price' else price,
                    predicted_quantity=predicted_quantity if prediction_type == 'quantity' else quantity,
                    algorithm_used=algorithm
                )
                
                # Generate charts
                charts['comparison'] = ml_predictor.get_model_comparison_chart()
                charts['price_distribution'] = ml_predictor.get_price_distribution_chart()
                charts['quantity_vs_price'] = ml_predictor.get_quantity_vs_price_chart()
                
            except Exception as e:
                messages.error(request, f'Prediction error: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors in the form.')
    else:
        form = MLPredictionForm()
    
    context = {
        'form': form,
        'prediction_result': prediction_result,
        'charts': charts,
        'user_type': request.session.get('user_type'),
    }
    return render(request, 'pharmacy_app/ml_prediction.html', context)


@login_required
def substitution_view(request):
    if 'customer_id' not in request.session or request.session.get('user_type') != 'customer':
        messages.error(request, 'Please login as customer to access this page.')
        return redirect('customer_login')

    results = None
    picked = None
    if request.method == 'POST':
        form = MedicineSubstitutionForm(request.POST)
        if form.is_valid():
            picked = form.cleaned_data['medicine']
            top_k = form.cleaned_data['top_k']
            try:
                results = substitution_recommender.recommend_substitutes(picked, top_k=top_k)
                customer = get_object_or_404(Customer, cust_id=request.session['customer_id'])
                ModelSearchHistory.objects.create(
                    customer=customer,
                    query_type='substitution',
                    input_text='',
                    selected_medicine=picked,
                    results={'recommendations': results}
                )
                for r in results:
                    try:
                        med = Medicine.objects.get(id=r['id'])
                        SubstitutionRecommendation.objects.create(
                            customer=customer,
                            source_medicine=picked,
                            recommended_medicine=med,
                            similarity=r.get('similarity', 0.0),
                            price_difference=(picked.m_price - med.m_price) if picked and med else None
                        )
                    except Medicine.DoesNotExist:
                        continue
            except Exception as e:
                messages.error(request, f'Recommendation error: {str(e)}')
    else:
        form = MedicineSubstitutionForm()

    return render(request, 'pharmacy_app/substitution.html', {
        'form': form,
        'results': results,
        'picked': picked,
        'user_type': request.session.get('user_type'),
    })


@login_required
def disease_recommendation_view(request):
    if 'customer_id' not in request.session or request.session.get('user_type') != 'customer':
        messages.error(request, 'Please login as customer to access this page.')
        return redirect('customer_login')

    results = None
    query = None
    if request.method == 'POST':
        form = DiseaseRecommendationForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['disease_text']
            top_k = form.cleaned_data['top_k']
            try:
                results = disease_recommender.recommend(query, top_k=top_k)
                customer = get_object_or_404(Customer, cust_id=request.session['customer_id'])
                ModelSearchHistory.objects.create(
                    customer=customer,
                    query_type='disease',
                    input_text=query,
                    selected_medicine=None,
                    results={'recommendations': results}
                )
                for r in results:
                    try:
                        med = Medicine.objects.get(id=r['id'])
                        DiseaseRecommendation.objects.create(
                            customer=customer,
                            query_text=query,
                            recommended_medicine=med,
                            similarity=r.get('similarity', 0.0)
                        )
                    except Medicine.DoesNotExist:
                        continue
            except Exception as e:
                messages.error(request, f'Recommendation error: {str(e)}')
    else:
        form = DiseaseRecommendationForm()

    return render(request, 'pharmacy_app/disease_recommendation.html', {
        'form': form,
        'results': results,
        'query': query,
        'user_type': request.session.get('user_type'),
    })

@login_required
def stock_check(request):
    """Stock check view for admins"""
    if 'admin_id' not in request.session or request.session.get('user_type') != 'admin':
        messages.error(request, 'Please login as admin to access this page.')
        return redirect('admin_login')
    
    medicines = Medicine.objects.all().order_by('m_name')
    total_medicines = Medicine.objects.count()
    low_stock_medicines = Medicine.objects.filter(m_quantity__lt=10)
    expired_medicines = Medicine.objects.filter(m_edate__lt=date.today())
    user_type=request.session.get('user_type')
    context={
        "medicines":medicines,
        "available":total_medicines,
        "low_stock_medicines":low_stock_medicines,
        "expired_medicine":expired_medicines,
        'user_type': user_type
    }
    return render(request, 'pharmacy_app/stock_check.html',context)

@login_required
def add_medicine(request):
    """Add medicine view for admins"""
    if 'admin_id' not in request.session or request.session.get('user_type') != 'admin':
        messages.error(request, 'Please login as admin to access this page.')
        return redirect('admin_login')
    
    if request.method == 'POST':
        form = MedicineForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Medicine added successfully!')
            return redirect('stock_check')
    else:
        form = MedicineForm()
    
    return render(request, 'pharmacy_app/add_medicine.html', {'form': form, 'user_type': request.session.get('user_type')})

@login_required
def remove_medicine(request):
    """Remove medicine view for admins"""
    if 'admin_id' not in request.session or request.session.get('user_type') != 'admin':
        messages.error(request, 'Please login as admin to access this page.')
        return redirect('admin_login')
    
    if request.method == 'POST':
        medicine_id = request.POST.get('medicine_id')
        if medicine_id:
            medicine = get_object_or_404(Medicine, id=medicine_id)
            medicine_name = medicine.m_name
            medicine.delete()
            messages.success(request, f'{medicine_name} removed successfully!')
            return redirect('stock_check')
    
    medicines = Medicine.objects.all()
    return render(request, 'pharmacy_app/remove_medicine.html', {'medicines': medicines, 'user_type': request.session.get('user_type')})

@login_required
def update_quantity(request):
    """Update medicine quantity view for admins"""
    if 'admin_id' not in request.session or request.session.get('user_type') != 'admin':
        messages.error(request, 'Please login as admin to access this page.')
        return redirect('admin_login')
    
    if request.method == 'POST':
        medicine_id = request.POST.get('medicine_id')
        new_quantity = request.POST.get('new_quantity')
        
        if medicine_id and new_quantity:
            try:
                medicine = get_object_or_404(Medicine, id=medicine_id)
                old_quantity = medicine.m_quantity
                medicine.m_quantity = int(new_quantity)
                medicine.save()
                messages.success(request, f'Quantity updated for {medicine.m_name} from {old_quantity} to {new_quantity}')
                return redirect('update_quantity')
            except ValueError:
                messages.error(request, 'Invalid quantity value')
    
    medicines = Medicine.objects.all()
    return render(request, 'pharmacy_app/update_quantity.html', {'medicines': medicines, 'user_type': request.session.get('user_type')})

@login_required
def check_requests(request):
    """Check customer requests view for admins"""
    if 'admin_id' not in request.session or request.session.get('user_type') != 'admin':
        messages.error(request, 'Please login as admin to access this page.')
        return redirect('admin_login')
    
    if request.method == 'POST':
        request_id = request.POST.get('request_id')
        action = request.POST.get('action')
        
        if request_id and action:
            try:
                request_item = get_object_or_404(RequestList, id=request_id)
                if action == 'complete':
                    request_item.status = 'Completed'
                    messages.success(request, f'Request for {request_item.med_name} marked as completed')
                elif action == 'reject':
                    request_item.status = 'Rejected'
                    messages.success(request, f'Request for {request_item.med_name} rejected')
                request_item.save()
            except Exception as e:
                messages.error(request, f'Error processing request: {str(e)}')
    
    requests = RequestList.objects.all().order_by('-request_date')
    pending_count = RequestList.objects.filter(status='Pending').count()
    completed_count = RequestList.objects.filter(status='Completed').count()
    rejected_count = RequestList.objects.filter(status='Rejected').count()
    total_count = RequestList.objects.count()
    
    context = {
        'requests': requests,
        'pending_count': pending_count,
        'completed_count': completed_count,
        'rejected_count': rejected_count,
        'total_count': total_count,
        'user_type': request.session.get('user_type'),
    }
    return render(request, 'pharmacy_app/check_requests.html', context)

@login_required
def generate_bill(request):
    """Generate bill view for customers"""
    if 'customer_id' not in request.session or request.session.get('user_type') != 'customer':
        messages.error(request, 'Please login as customer to access this page.')
        return redirect('customer_login')
    
    customer = get_object_or_404(Customer, cust_id=request.session['customer_id'])
    last_purchase_id = request.session.pop('last_purchase_id', None)
    if last_purchase_id:
        purchases = Purchase.objects.filter(id=last_purchase_id, customer=customer)
    else:
        purchases = Purchase.objects.filter(customer=customer).order_by('-purchase_date')
    
    if not purchases.exists():
        messages.warning(request, 'No purchases found to generate bill.')
        return redirect('customer_dashboard')
    
    total_amount = sum(purchase.total_price for purchase in purchases)
    gst_amount = total_amount * Decimal('0.04')  # 4% GST
    final_amount = total_amount + gst_amount
    
    context = {
        'customer': customer,
        'purchases': purchases,
        'total_amount': total_amount,
        'gst_amount': gst_amount,
        'final_amount': final_amount,
        'bill_date': timezone.now(),
        'user_type': request.session.get('user_type'),
    }
    
    return render(request, 'pharmacy_app/bill.html', context)

@login_required
def generate_report(request):
    """Generate report view for admins"""
    if 'admin_id' not in request.session or request.session.get('user_type') != 'admin':
        messages.error(request, 'Please login as admin to access this page.')
        return redirect('admin_login')
    
    # Get all medicines
    medicines = Medicine.objects.all()
    
    # Get sales data
    sales = Purchase.objects.all().order_by('-purchase_date')
    total_sales = sales.aggregate(Sum('total_price'))['total_price__sum'] or 0
    
    # Get requests data
    requests = RequestList.objects.all()
    pending_requests = requests.filter(status='Pending').count()
    completed_requests = requests.filter(status='Completed').count()
    
    context = {
        'medicines': medicines,
        'sales': sales,
        'total_sales': total_sales,
        'pending_requests': pending_requests,
        'completed_requests': completed_requests,
        'user_type': request.session.get('user_type'),
    }
    
    return render(request, 'pharmacy_app/report.html', context)

def about(request):
    """About page view"""
    return render(request, 'pharmacy_app/about.html')

def contact(request):
    """Contact page view"""
    return render(request, 'pharmacy_app/contact.html')

@csrf_exempt
def payment_page(request):
    """Payment page for confirming purchase and selecting payment method"""
    if 'customer_id' not in request.session or request.session.get('user_type') != 'customer':
        messages.error(request, 'Please login as customer to access this page.')
        return redirect('customer_login')

    # Get purchase details from session
    medicine_id = request.session.get('pending_medicine_id')
    quantity = request.session.get('pending_quantity')
    if not medicine_id or not quantity:
        messages.error(request, 'No pending purchase found.')
        return redirect('buy_medicine')

    medicine = get_object_or_404(Medicine, id=medicine_id)
    total_price = medicine.m_price * int(quantity)

    if request.method == 'POST':
        payment_method = request.POST.get('payment_method')
        payment_reference = ''
        # Validate and simulate payment
        if payment_method == 'Card':
            card_number = request.POST.get('card_number', '').replace(' ', '')
            card_expiry = request.POST.get('card_expiry', '')
            card_cvv = request.POST.get('card_cvv', '')
            card_name = request.POST.get('card_name', '')
            if not (card_number.isdigit() and 16 <= len(card_number) <= 19):
                messages.error(request, 'Invalid card number.')
                return redirect('payment_page')
            if not card_expiry or len(card_expiry) != 5 or card_expiry[2] != '/':
                messages.error(request, 'Invalid expiry date.')
                return redirect('payment_page')
            if not (card_cvv.isdigit() and 3 <= len(card_cvv) <= 4):
                messages.error(request, 'Invalid CVV.')
                return redirect('payment_page')
            if len(card_name) < 2:
                messages.error(request, 'Invalid cardholder name.')
                return redirect('payment_page')
            # Simulate payment gateway
            if randint(1, 10) == 1:
                messages.error(request, 'Payment failed. Please try again.')
                return redirect('payment_page')
            payment_reference = f"CARD-{card_number[-4:]}-{randint(1000,9999)}"
        elif payment_method == 'UPI':
            upi_id = request.POST.get('upi_id', '')
            if not upi_id or '@' not in upi_id:
                messages.error(request, 'Invalid UPI ID.')
                return redirect('payment_page')
            # Simulate payment gateway
            if randint(1, 10) == 1:
                messages.error(request, 'Payment failed. Please try again.')
                return redirect('payment_page')
            payment_reference = f"UPI-{upi_id.split('@')[0]}-{randint(1000,9999)}"
        elif payment_method == 'Cash':
            payment_reference = f"CASH-{randint(1000,9999)}"
        else:
            messages.error(request, 'Invalid payment method.')
            return redirect('payment_page')
        customer = get_object_or_404(Customer, cust_id=request.session['customer_id'])
        # Create purchase with payment info
        purchase = Purchase.objects.create(
            customer=customer,
            medicine=medicine,
            quantity=quantity,
            total_price=total_price,
            payment_method=payment_method,
            payment_reference=payment_reference
        )
        medicine.m_quantity -= int(quantity)
        medicine.save()
        del request.session['pending_medicine_id']
        del request.session['pending_quantity']
        request.session['last_purchase_id'] = purchase.id
        messages.success(request, f'Payment successful! You purchased {quantity} units of {medicine.m_name}.')
        return redirect('generate_bill')

    context = {
        'medicine': medicine,
        'quantity': quantity,
        'total_price': total_price,
        'user_type': request.session.get('user_type'),
    }
    return render(request, 'pharmacy_app/payment.html', context)