# Render Deployment Guide

## Prerequisites
- GitHub account with your project repository
- Render account (sign up at https://render.com)

## Deployment Steps

### Option 1: Using render.yaml (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```

2. **Connect to Render**
   - Go to https://dashboard.render.com
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and create both services

3. **Set Environment Variables**
   - Go to Backend service → Environment
   - Add: `GROQ_API_KEY` = your_groq_api_key
   - Save changes

### Option 2: Manual Setup

#### Backend Deployment

1. **Create Web Service**
   - Dashboard → New + → Web Service
   - Connect your GitHub repository
   - Configure:
     - Name: `multi-disease-backend`
     - Region: Oregon (US West)
     - Branch: `main`
     - Root Directory: Leave empty
     - Runtime: Python 3
     - Build Command: `cd backend && pip install -r requirements.txt`
     - Start Command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`

2. **Environment Variables**
   - Add: `PYTHON_VERSION` = `3.10.0`
   - Add: `GROQ_API_KEY` = your_groq_api_key

3. **Health Check**
   - Path: `/health`

#### Frontend Deployment

1. **Create Static Site**
   - Dashboard → New + → Static Site
   - Connect your GitHub repository
   - Configure:
     - Name: `multi-disease-frontend`
     - Branch: `main`
     - Root Directory: Leave empty
     - Build Command: `cd frontend && npm install && npm run build`
     - Publish Directory: `frontend/build`

2. **Environment Variables**
   - Add: `REACT_APP_API_URL` = `https://your-backend-url.onrender.com`
   - Replace with your actual backend URL from step 1

3. **Rewrite Rules**
   - Add rule: `/*` → `/index.html` (for React Router)

## Important Notes

1. **Free Tier Limitations**
   - Services spin down after 15 minutes of inactivity
   - First request after spin-down takes ~30 seconds
   - 750 hours/month free

2. **Database**
   - SQLite database will reset on each deployment
   - For production, consider upgrading to PostgreSQL

3. **CORS Configuration**
   - Backend already configured to accept requests from frontend
   - Update CORS origins in `backend/main.py` if needed

4. **Custom Domain** (Optional)
   - Go to Settings → Custom Domain
   - Add your domain and configure DNS

## Troubleshooting

### Backend won't start
- Check logs in Render dashboard
- Verify all dependencies in `requirements.txt`
- Ensure `GROQ_API_KEY` is set

### Frontend can't connect to backend
- Verify `REACT_APP_API_URL` is set correctly
- Check CORS settings in backend
- Ensure backend service is running

### Database errors
- Database resets on deployment (SQLite limitation)
- Run `setup_demo.py` to populate demo data
- Consider PostgreSQL for persistent data

## Post-Deployment

1. **Test the application**
   - Visit your frontend URL
   - Create a test patient
   - Add health records
   - Generate predictions

2. **Monitor logs**
   - Backend: Check for any errors
   - Frontend: Check browser console

3. **Set up demo data** (Optional)
   - SSH into backend service
   - Run: `python setup_demo.py`

## URLs
- Backend API: `https://multi-disease-backend.onrender.com`
- Frontend: `https://multi-disease-frontend.onrender.com`
- API Docs: `https://multi-disease-backend.onrender.com/docs`

## Support
For issues, check:
- Render documentation: https://render.com/docs
- Project logs in Render dashboard
- GitHub repository issues
