FROM continuumio/miniconda3

# ADD ./ndcontainers /opt/ndcontainers
ADD ./requirements.txt /tmp/requirements.txt
RUN conda install --file /tmp/requirements.txt
