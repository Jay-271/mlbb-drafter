# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the trained models folder
COPY expir.py ./  
COPY db.py ./
COPY ./Trained_Models/ /app/Trained_Models
COPY ./mlbb_work/ /app/mlbb_work

# Copy requirements and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port Cloud Run uses (default 8080)
EXPOSE 8080

# Use Gunicorn to serve the app (assuming your Flask app is defined in expir.py as "app")
CMD ["gunicorn", "-b", "0.0.0.0:8080", "expir:app"]
