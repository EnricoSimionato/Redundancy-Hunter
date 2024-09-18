FROM python
# Setting the PYTHONPATH
ENV PYTHONPATH=/Redundancy-Hunter/src
# Setting
ENV HUGGINGFACE_HUB_TOKEN=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
# Setting the working directory
WORKDIR /Redundancy-Hunter
# Copying the entire contents of the Redundancy-Hunter directory
COPY . .
# Installing dependencies
RUN pip install -r requirements.txt
# Setting the command to run the script
CMD ["python3", "src/redhunter/analysis_launcher.py", "CONFIG_SERVER.yaml"]

