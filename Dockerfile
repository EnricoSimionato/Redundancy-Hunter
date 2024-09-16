FROM python
WORKDIR /Redundancy-Hunter
COPY src/ src/
COPY test.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir output
VOLUME /Redundancy-Hunter/output
CMD ["python3", "./test.py"]