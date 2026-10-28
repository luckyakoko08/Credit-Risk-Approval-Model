# Use a lightweight official Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install the required dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your trained model pipeline and the FastAPI app code
COPY credit_risk_pipeline.pkl .
COPY app.py .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
