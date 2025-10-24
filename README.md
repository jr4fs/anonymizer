# Setup Instructions 

These instructions assume you are **always using the Presidio analyzer** (`presidio_filtered.enable: true`) for additional rule-based entity detection.  
Presidio uses spaCy locally

---

## ⚙️ 1. Create and Activate the Conda Environment

```bash
# Create the environment from the provided YAML
conda env create -f environment.yml

# Activate it
conda activate casenote-anon

# Download the spaCy language model required by Presidio
python -m spacy download en_core_web_lg

python anonymize.py --config config.yaml
