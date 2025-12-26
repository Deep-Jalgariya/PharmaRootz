# 🏥 PharmaRootz - Advanced Pharmacy Management System

[![Django](https://img.shields.io/badge/Django-5.2.4-green.svg)](https://djangoproject.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![AI Powered](https://img.shields.io/badge/AI-Google%20Gemini-orange.svg)](https://ai.google.dev/)


A comprehensive pharmacy management system with an integrated **AI chatbot** powered by Google Gemini. Built with Django, featuring modern UI/UX and intelligent pharmacy assistance.

## 🌟 Key Features

### 💊 **Pharmacy Management**
- **Inventory Management** - Track medications, stock levels, expiry dates
- **Customer Management** - Patient records, prescription history
- **Admin Dashboard** - Complete pharmacy operations control
- **User Authentication** - Secure login system for customers and admins
- **Responsive Design** - Works perfectly on all devices

### 🤖 **AI-Powered Chatbot**
- **Google Gemini Integration** - Advanced AI for pharmacy queries
- **Floating Widget** - Always accessible on every page
- **Real-time Responses** - Instant AI assistance
- **Pharmacy-Focused** - Specialized for healthcare and medication queries
- **Mobile Optimized** - Perfect experience on smartphones

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Django 5.2+
- Google Gemini API Key
- Internet connection

### 1. Clone & Setup
```bash
git clone <repository-url>
cd pharma-rootz
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key
1. Visit: https://makersuite.google.com/app/apikey
2. Create API Key
3. Copy the key 



### 3. Run the Application
```bash
# Test setup
python test_chatbot.py

# Start server
python manage.py runserver
```

### 4 . Access the System
- **Main Site**: http://127.0.0.1:8000/
- **Admin Panel**: http://127.0.0.1:8000/admin/
- **AI Chatbot**: Look for green floating button (bottom-right)

## 📱 Screenshots

### WhatsApp-Style AI Chatbot
```
┌─────────────────────────────────┐
│ 🤖 Pharma Assistant    ● Online │
├─────────────────────────────────┤
│  🏥 Hi! I'm your Pharma        │
│     Assistant. I can help with: │
│     • Medication information    │
│     • Drug interactions         │
│     • Health advice            │
│     • Pharmacy services        │
│                                 │
│                    What is      │
│                    paracetamol? │
│                         ┌─────┐ │
│  💊 Paracetamol is a common    │
│     pain reliever and fever    │
│     reducer. It's used for...  │
│                                 │
├─────────────────────────────────┤
│ Type a message...          📤  │
└─────────────────────────────────┘
```

### Main Dashboard
- Clean, modern interface
- Healthcare-focused design
- Responsive navigation
- Professional styling

## 🏗️ Project Structure

```
pharma-rootz/
├── 📁 pharmacy_system/          # Main Django project
│   ├── settings.py              # Project settings
│   ├── urls.py                  # URL routing
│   └── wsgi.py                  # WSGI config
│
├── 📁 pharmacy_app/             # Core pharmacy functionality
│   ├── models.py                # Database models
│   ├── views.py                 # Business logic
│   ├── urls.py                  # App URLs
│   └── admin.py                 # Admin interface
│
├── 📁 chatbot/                  # AI Chatbot app
│   ├── views.py                 # Chat API & logic
│   ├── urls.py                  # Chat routing
│   └── models.py                # Chat models (if needed)
│
├── 📁 templates/                # HTML templates
│   ├── base.html                # Base template with chatbot
│   ├── 📁 pharmacy_app/         # Pharmacy templates
│   └── 📁 chatbot/              # Chat templates
│
├── 📁 static/                   # Static files
│   ├── 📁 css/                  # Stylesheets
│   ├── 📁 js/                   # JavaScript
│   └── 📁 images/               # Images & logos
│
├── 📄 requirements.txt          # Python dependencies
├── 📄 manage.py                 # Django management
├── 📄 test_chatbot.py          # Chatbot testing
├── 📄 check_gemini_models.py   # Model checker
└── 📄 README.md                # This file
```

## 🔧 Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your-google-gemini-api-key
DEBUG=True                       # Set to False in production
SECRET_KEY=your-secret-key       # Change in production
```

### Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### Static Files
```bash
python manage.py collectstatic
```

## 🤖 AI Chatbot Features

### Intelligent Responses
- **Medication Information** - Drug details, uses, dosages
- **Drug Interactions** - Safety warnings and combinations
- **Health Advice** - General wellness guidance
- **Pharmacy Services** - Store information and procedures

### Technical Features
- **Real-time Chat** - No page reloads required
- **Typing Indicators** - Shows when AI is thinking
- **Error Handling** - Graceful failure management
- **Mobile Responsive** - Perfect on all screen sizes
- **CSRF Protection** - Secure API communications
- **Model Auto-Selection** - Uses best available Gemini model

### Customization Options
```python
# Modify AI behavior in chatbot/views.py
pharmacy_prompt = f"""
Your custom instructions here...
User question: {user_message}
"""

# Change chatbot colors in templates/base.html
background: #25D366;  # WhatsApp Green
```

## 🛠️ Development

### Adding New Features
1. **Pharmacy Features** - Add to `pharmacy_app/`
2. **Chatbot Enhancements** - Modify `chatbot/views.py`
3. **UI Changes** - Update templates and static files
4. **API Extensions** - Add new endpoints in `urls.py`

### Testing
```bash
# Test chatbot functionality
python test_chatbot.py

# Check available AI models
python check_gemini_models.py

# Run Django tests
python manage.py test
```

### Deployment Checklist
- [ ] Set `DEBUG = False`
- [ ] Configure production database
- [ ] Set up static file serving
- [ ] Configure HTTPS
- [ ] Set secure environment variables
- [ ] Test chatbot in production

## 📚 API Documentation

### Chatbot API
```http
POST /chatbot/chat/
Content-Type: application/json

{
    "message": "What is aspirin used for?"
}
```

**Response:**
```json
{
    "success": true,
    "response": "Aspirin is commonly used for..."
}
```

### Error Handling
```json
{
    "success": false,
    "error": "Error message here"
}
```

## 🔒 Security Features

- **CSRF Protection** - All forms protected
- **Input Validation** - Sanitized user inputs
- **API Key Security** - Environment variable storage
- **User Authentication** - Secure login system
- **SQL Injection Prevention** - Django ORM protection

## 🌐 Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+


## 📱 Mobile Features

- **Responsive Design** - Adapts to all screen sizes
- **Touch Optimized** - Perfect for mobile interaction
- **Fast Loading** - Optimized for mobile networks
- **Offline Graceful** - Handles connection issues


### Development Guidelines
- Follow Django best practices
- Write tests for new features
- Update documentation
- Ensure mobile compatibility
- Test chatbot functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Django Framework** - Web framework
- **Google Gemini** - AI capabilities
- **Bootstrap** - UI components
- **Font Awesome** - Icons
- **WhatsApp** - Chat interface inspiration



## 🎯 Quick Commands

```bash
# Setup and run
pip install -r requirements.txt
set GEMINI_API_KEY=your-key-here
python test_chatbot.py
python manage.py runserver

# Development
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py collectstatic


```

---

**Built with ❤️ for modern pharmacy management and AI-powered customer assistance.**

*PharmaRootz - Where Technology Meets Healthcare* 🏥✨