# Use an official Python runtime as a parent image
FROM dockerhub.kubekey.local/kubesphereio/torch-base:2024-01-12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY service.py /app/service.py
COPY config.py /app/config.py
COPY id_text_map.json /app/id_text_map.json
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run service.py when the container launches
CMD ["python", "service.py"]