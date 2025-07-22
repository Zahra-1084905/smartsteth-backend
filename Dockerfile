# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first, to cache Docker layers
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Run the Flask app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
