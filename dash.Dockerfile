FROM python:3.11.5-bookworm

WORKDIR /root

RUN pip3 install dash
RUN pip3 install joblib
RUN pip3 install dash_bootstrap_components
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn==1.3.0

RUN pip3 install mlflow

RUN pip3 install dash[testing]
RUN pip3 install pytest
RUN pip3 install pytest-depends

COPY ./app /root/

CMD tail -f /dev/null
