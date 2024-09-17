FROM python
ENV PYTHONPATH=/Redundancy-Hunter/src
WORKDIR /Redundancy-Hunter
COPY src/ src/
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN git init
CMD ["python3", "src/redhunter/analysis_launcher.py", "CONFIG_SERVER.yaml"]