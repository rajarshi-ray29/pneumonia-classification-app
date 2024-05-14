# Use the official Python image with Python 3.9
FROM python:3.9

# Install HDF5 library (assuming it's still needed)
RUN apt-get update && \
    apt-get install -y libhdf5-dev

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose any port you'll use for communication if needed
EXPOSE 8080 

# Define the default command using the args from docker-compose
CMD ["python", "image_classifier.py", "/app/input", "/app/output/results.json"]
