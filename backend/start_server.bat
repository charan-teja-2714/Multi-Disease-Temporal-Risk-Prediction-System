@echo off
echo Setting up Multi-Disease RAG Pipeline
echo.

REM Check if GROQ_API_KEY is set
if "%GROQ_API_KEY%"=="" (
    echo ERROR: GROQ_API_KEY environment variable not set
    echo.
    echo To enable RAG functionality:
    echo 1. Get API key from: https://console.groq.com/
    echo 2. Set environment variable: set GROQ_API_KEY=your_key_here
    echo 3. Run this script again
    echo.
    echo Starting server without RAG functionality...
    echo.
) else (
    echo ✓ GROQ_API_KEY found - RAG functionality enabled
    echo.
)

echo Starting FastAPI server...
python main.py