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
        
        # Create the model
        model = genai.GenerativeModel("gemini-2.5-flash")
        
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

