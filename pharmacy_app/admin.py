from django.contrib import admin
from .models import Customer, Admin, Medicine, Purchase, RequestList, MLPrediction, ModelSearchHistory, SubstitutionRecommendation, DiseaseRecommendation

@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ('cust_id', 'c_fname', 'c_lname', 'c_email', 'c_phone_no', 'created_at')
    search_fields = ('cust_id', 'c_fname', 'c_lname', 'c_email')
    list_filter = ('c_gender', 'created_at')
    readonly_fields = ('cust_id', 'created_at')

@admin.register(Admin)
class AdminAdmin(admin.ModelAdmin):
    list_display = ('admin_id', 'a_fname', 'a_lname', 'a_email', 'a_phone_no', 'created_at')
    search_fields = ('admin_id', 'a_fname', 'a_lname', 'a_email')
    list_filter = ('a_gender', 'created_at')
    readonly_fields = ('admin_id', 'created_at')

@admin.register(Medicine)
class MedicineAdmin(admin.ModelAdmin):
    list_display = ('m_name', 'm_price', 'm_quantity', 'm_mdate', 'm_edate', 'is_generic', 'is_expired')
    search_fields = ('m_name', 'm_descr', 'composition', 'indications')
    list_filter = ('m_mdate', 'm_edate', 'created_at')
    readonly_fields = ('created_at',)
    
    def is_expired(self, obj):
        return obj.is_expired
    is_expired.boolean = True
    is_expired.short_description = 'Expired'

@admin.register(Purchase)
class PurchaseAdmin(admin.ModelAdmin):
    list_display = ('customer', 'medicine', 'quantity', 'total_price', 'purchase_date')
    search_fields = ('customer__cust_id', 'medicine__m_name')
    list_filter = ('purchase_date',)
    readonly_fields = ('purchase_date',)

@admin.register(RequestList)
class RequestListAdmin(admin.ModelAdmin):
    list_display = ('c_id', 'med_name', 'med_quan', 'status', 'request_date')
    search_fields = ('c_id__cust_id', 'med_name')
    list_filter = ('status', 'request_date')
    readonly_fields = ('request_date',)

@admin.register(MLPrediction)
class MLPredictionAdmin(admin.ModelAdmin):
    list_display = ('customer', 'medicine_name', 'predicted_price', 'predicted_quantity', 'algorithm_used', 'prediction_date')
    search_fields = ('customer__cust_id', 'medicine_name', 'algorithm_used')
    list_filter = ('algorithm_used', 'prediction_date')
    readonly_fields = ('prediction_date',)

@admin.register(ModelSearchHistory)
class ModelSearchHistoryAdmin(admin.ModelAdmin):
    list_display = ('customer', 'query_type', 'selected_medicine', 'created_at')
    search_fields = ('customer__cust_id', 'input_text', 'selected_medicine__m_name')
    list_filter = ('query_type', 'created_at')
    readonly_fields = ('created_at',)

@admin.register(SubstitutionRecommendation)
class SubstitutionRecommendationAdmin(admin.ModelAdmin):
    list_display = ('customer', 'source_medicine', 'recommended_medicine', 'similarity', 'price_difference', 'created_at')
    search_fields = ('customer__cust_id', 'source_medicine__m_name', 'recommended_medicine__m_name')
    list_filter = ('created_at',)
    readonly_fields = ('created_at',)

@admin.register(DiseaseRecommendation)
class DiseaseRecommendationAdmin(admin.ModelAdmin):
    list_display = ('customer', 'query_text', 'recommended_medicine', 'similarity', 'created_at')
    search_fields = ('customer__cust_id', 'query_text', 'recommended_medicine__m_name')
    list_filter = ('created_at',)
    readonly_fields = ('created_at',)
