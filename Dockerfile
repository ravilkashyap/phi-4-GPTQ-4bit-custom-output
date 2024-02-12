FROM nvidia/cuda:12.2.0-base-ubuntu20.04

WORKDIR /app
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install fastapi==0.101.0
RUN pip3 install pydantic==2.1.1
RUN pip3 install pydantic_core==2.4.0
RUN pip3 install urllib3==2.0.4
RUN pip3 install uvicorn==0.23.2
RUN pip3 install transformers
RUN pip3 install torch==2.0.1
RUN pip3 install accelerate
COPY ./main.py /app/main.py

EXPOSE 7000
ENV HOST 0.0.0.0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--reload"]
