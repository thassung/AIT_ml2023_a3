FROM python:3.11.5-bookworm

WORKDIR /root

RUN pip3 install dash
RUN pip3 install joblib
RUN pip3 install dash_bootstrap_components
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install mlflow
# EXPOSE 5000

# COPY ./app /root

CMD tail -f /dev/null
