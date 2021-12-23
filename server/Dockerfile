FROM ocr:base

COPY . /workdir
WORKDIR /workdir
ENV PYTHONPATH=/workdir
ENV PYTHONUNBUFFERED=1

RUN /bin/bash \
    && pwd 

CMD ["bash", "scripts/start.sh"]