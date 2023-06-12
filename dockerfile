# Use an official Python runtime as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /plates

# Copy the application code to the container
COPY . .

# Activate a virtual environment
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"


# Run the application
CMD ["python", "main.py"]
