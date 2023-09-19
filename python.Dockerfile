FROM python:3.11.5-bookworm

RUN pip3 install --upgrade pip
RUN pip3 install ipykernel
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install mlflow
RUN pip3 install ppscore

CMD tail -f /dev/null

