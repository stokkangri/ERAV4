#!/usr/bin/env python3
"""
Test script for the Animal & File Upload Application
"""

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test if all required files and directories exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        'app.py',
        'templates/index.html',
        'static/style.css',
        'static/script.js',
        'static/images/cat.jpg',
        'static/images/dog.jpg',
        'static/images/elephant.jpg',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'templates',
        'static',
        'static/images',
        'uploads'
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_good = False
    
    return all_good

def test_flask_import():
    """Test if Flask application can be imported"""
    print("\n🔍 Testing Flask import...")
    try:
        from app import app
        print("✅ Flask application imported successfully")
        return True
    except Exception as e:
        print(f"❌ Flask import failed: {e}")
        return False

def test_dependencies():
    """Test if required Python packages are installed"""
    print("\n🔍 Testing dependencies...")
    try:
        import flask
        import werkzeug
        print(f"✅ Flask {flask.__version__}")
        print(f"✅ Werkzeug {werkzeug.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Dependency missing: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Animal & File Upload Application\n")
    
    tests = [
        test_file_structure,
        test_flask_import,
        test_dependencies
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*50)
    if all(results):
        print("🎉 All tests passed! Application is ready to run.")
        print("\n🚀 To start the application, run:")
        print("   python run.py")
        print("\n📱 Then open your browser to: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
