FROM python:3.13-slim
RUN apt-get update && apt-get install -y git
# Setting the PYTHONPATH
ENV PYTHONPATH=/Redundancy-Hunter/src
# Setting the working directory
WORKDIR /Redundancy-Hunter
# Copying the entire contents of the Redundancy-Hunter directory
COPY . .
# Installing dependencies
RUN pip install -r requirements.txt
# Logging in to the Hugging Face model hub
RUN huggingface-cli login --token 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
# Setting the command to run the script
CMD ["python3", "src/redhunter/analysis_launcher.py", "CONFIG_SERVER.yaml"]

