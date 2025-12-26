from django.urls import path
from . import views

urlpatterns = [
    # Public pages
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    
    # User type selection
    path('user-type/', views.user_type_selection, name='user_type_selection'),
    
    # Customer routes
    path('customer/register/', views.customer_register, name='customer_register'),
    path('customer/login/', views.customer_login, name='customer_login'),
    path('customer/dashboard/', views.customer_dashboard, name='customer_dashboard'),
    path('customer/buy-medicine/', views.buy_medicine, name='buy_medicine'),
    path('customer/show-medicines/', views.show_medicines, name='show_medicines'),
    path('customer/request-medicine/', views.request_medicine, name='request_medicine'),
    path('customer/ml-prediction/', views.ml_prediction, name='ml_prediction'),
    path('customer/substitution/', views.substitution_view, name='substitution'),
    path('customer/disease-recommendation/', views.disease_recommendation_view, name='disease_recommendation'),
    path('customer/bill/', views.generate_bill, name='generate_bill'),
    path('customer/payment/', views.payment_page, name='payment_page'),
        
    # Admin routes - Changed from 'admin/' to 'store-admin/'
    path('store-admin/register/', views.admin_register, name='admin_register'),
    path('store-admin/login/', views.admin_login, name='admin_login'),
    path('store-admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('store-admin/stock-check/', views.stock_check, name='stock_check'),
    path('store-admin/add-medicine/', views.add_medicine, name='add_medicine'),
    path('store-admin/remove-medicine/', views.remove_medicine, name='remove_medicine'),
    path('store-admin/update-quantity/', views.update_quantity, name='update_quantity'),
    path('store-admin/check-requests/', views.check_requests, name='check_requests'),
    path('store-admin/generate-report/', views.generate_report, name='generate_report'),
    
    # Logout
    path('logout/', views.logout_view, name='logout'),
]