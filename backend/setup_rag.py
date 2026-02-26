#!/usr/bin/env python3
"""
Environment setup script for RAG pipeline
"""

import os
import sys

def setup_environment():
    """Setup environment for RAG pipeline"""
    
    print("Multi-Disease RAG Pipeline Setup")
    print("=" * 40)
    
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        return False
    
    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor}")
    
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'pydantic',
        'torch', 'numpy', 'pandas',
        'langchain', 'langchain_groq', 'pdfplumber', 'pytesseract', 'Pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"OK: {package}")
        except ImportError:
            print(f"MISSING: {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nInstall missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("OK: GROQ_API_KEY configured")
    else:
        print("WARNING: GROQ_API_KEY not set")
        print("   Get API key from: https://console.groq.com/")
        print("   Set it with: export GROQ_API_KEY='your_key_here'")
    
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("OK: Tesseract OCR available")
    except Exception:
        print("WARNING: Tesseract OCR not found")
        print("   Install: https://github.com/tesseract-ocr/tesseract")
    
    print("\n" + "=" * 40)
    print("Setup complete!")
    
    if groq_key:
        print("\nReady to start:")
        print("   python main.py")
    else:
        print("\nSet GROQ_API_KEY to enable RAG extraction")
    
    return True

if __name__ == "__main__":
    setup_environment()
