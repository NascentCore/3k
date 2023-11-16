FROM ubuntu:20.04
WORKDIR /infiniband-exporter
RUN apt-get update --fix-missing && apt-get install -y python3 pip infiniband-diags --fix-missing && \
    pip install prometheus-client==0.7.1
ADD https://github.com/guilbaults/infiniband-exporter/blob/master/infiniband-exporter.py ./
CMD python3 infiniband-exporter.py
