FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY peoplepulse peoplepulse
COPY pages pages
COPY sql sql
COPY data data
COPY Home.py .
EXPOSE 8501
# No DATABASE_URL -> demo mode on the bundled CSV. Set it to read from Postgres instead.
CMD ["streamlit", "run", "Home.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
