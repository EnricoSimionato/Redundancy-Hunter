FROM python
WORKDIR /Redundancy-Hunter
COPY src/ src/
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python3", "./src/redhunter/analysis_launcher.py", "CONFIG_LOCAL.yaml"]