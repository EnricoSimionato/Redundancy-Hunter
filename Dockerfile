#FROM python
#ENV PYTHONPATH=/Redundancy-Hunter/src
#WORKDIR /Redundancy-Hunter
#COPY src/ src/
#COPY requirements.txt .
#RUN pip install -r requirements.txt
#RUN git init
#CMD ["python3", "src/redhunter/analysis_launcher.py", "CONFIG_SERVER.yaml"]
FROM python

# Set the PYTHONPATH
ENV PYTHONPATH=/Redundancy-Hunter/src

# Set the working directory
WORKDIR /Redundancy-Hunter

# Copy the entire contents of the Redundancy-Hunter directory
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Set the command to run the script
CMD ["python3", "src/redhunter/analysis_launcher.py", "CONFIG_SERVER.yaml"]

