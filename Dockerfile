# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies, including Java
RUN apt-get update && \
    apt-get install -y poppler-utils tesseract-ocr libgl1-mesa-glx \
    default-jdk && \
    apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data files
RUN python -m nltk.downloader punkt averaged_perceptron_tagger stopwords

# Download and install the SpaCy language model
RUN python -m spacy download en_core_web_sm

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
