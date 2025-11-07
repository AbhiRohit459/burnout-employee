# 🚀 Deployment Guide

This guide will help you deploy the Employee Burnout Analysis Dashboard to Streamlit Cloud.

## Option 1: Streamlit Cloud (Recommended - Free & Easy)

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at https://streamlit.io/cloud)

### Steps:

1. **Ensure your code is on GitHub**
   - Your code is already pushed to: https://github.com/AbhiRohit459/burnout-employee.git
   - Make sure all files are committed and pushed

2. **Sign up for Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your repository: `AbhiRohit459/burnout-employee`
   - Set Main file path: `app.py`
   - Click "Deploy!"

4. **Wait for deployment**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt`
   - Your app will be live at: `https://burnout-employee.streamlit.app` (or similar)

### Important Notes:
- The Excel file (`employee_burnout_analysis-AI.xlsx`) must be in your repository
- Streamlit Cloud will automatically detect and use your `requirements.txt`
- Any changes pushed to the main branch will automatically redeploy

## Option 2: Docker Deployment

### Create Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and run:
```bash
docker build -t burnout-app .
docker run -p 8501:8501 burnout-app
```

## Option 3: Heroku

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. Deploy:
   ```bash
   heroku create burnout-app
   git push heroku main
   ```

## Option 4: Local Deployment

Run locally:
```bash
streamlit run app.py
```

Access at: http://localhost:8501

