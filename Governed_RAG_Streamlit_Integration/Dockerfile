FROM python:3.12-slim
WORKDIR /app
COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt
COPY . .
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000 8501
# default: the API. docker-compose overrides the command for the Streamlit UI.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
