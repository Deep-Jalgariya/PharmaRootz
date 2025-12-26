import json
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import google.generativeai as genai
from django.conf import settings


def chat_view(request):
    """Render the chat interface"""
    return render(request, 'chatbot/chat.html')


@require_http_methods(["POST"])
def chat_api(request):
    """Handle chat API requests"""
    try:
        # Parse JSON data from request
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({
                'success': False,
                'error': 'Message cannot be empty'
            })
        
        # Configure Gemini API
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if not api_key:
            return JsonResponse({
                'success': False,
                'error': 'Gemini API key not configured'
            })
        
        genai.configure(api_key=api_key)
        
        # Try to get available models and select the best one
        try:
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            
            # Preferred models in order of preference
            preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            model_name = None
            
            for preferred in preferred_models:
                for available in available_models:
                    if preferred in available:
                        model_name = available
                        break
                if model_name:
                    break
            
            # If no preferred model found, use first available
            if not model_name and available_models:
                model_name = available_models[0]
            
            # Fallback to default if nothing works
            if not model_name:
                model_name = 'gemini-1.5-flash'
                
        except Exception:
            # Fallback to default model if listing fails
            model_name = 'gemini-1.5-flash'
        
        # Create the model
        model = genai.GenerativeModel(model_name)
        
        # Create a pharmacy-focused prompt
        pharmacy_prompt = f"""
        You are a helpful pharmacy assistant AI. You provide information about:
        - Medications and their uses
        - Drug interactions and side effects
        - General health advice
        - Pharmacy services and procedures
        
        Please provide accurate, helpful information while always recommending users consult with healthcare professionals for medical advice.
        
        User question: {user_message}
        """
        
        # Generate response
        response = model.generate_content(pharmacy_prompt)
        
        if response.text:
            return JsonResponse({
                'success': True,
                'response': response.text
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'No response generated from AI'
            })
            
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }) 

