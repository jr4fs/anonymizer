# Setup Instructions 

These instructions assume you are **always using the Presidio analyzer** (`presidio_filtered.enable: true`) for additional rule-based entity detection.  
Presidio uses spaCy locally

---

## Clone repo
```bash
git clone https://github.com/jr4fs/anonymizer.git
```

## Create files
1. Put 'names.csv' file in data/
2. Put 'notes.csv' file in 'examples/'

## ⚙️ 1. Create and Activate the Conda Environment

```bash
# Create the environment from the provided YAML
conda env create -f environment.yml

# Activate it
conda activate casenote-anon

# Download the spaCy language model required by Presidio
python -m spacy download en_core_web_lg
```

## 2. Run the script
```bash
python anonymize.py --config config.yaml
```