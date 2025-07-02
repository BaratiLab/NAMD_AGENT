# Auto CGUI Agentic Pipeline

This project provides an agentic pipeline for automating CHARMM-GUI workflows and running molecular dynamics simulations using NAMD, with support for Retrieval-Augmented Generation (RAG) via LlamaIndex and Google Gemini.

## 1. Configuration

Before running the pipeline, update the configuration file:

Edit `auto_cgui_master/config.yml` with your CHARMM-GUI credentials and browser type. Example:

```yaml
BASE_URL: https://charmm-gui.org/
CGUSER: your cgui email goes here
CGPASS: your plain text password goes here
BROWSER_TYPE: firefox
```

- `BASE_URL`: URL for CHARMM-GUI (default: https://charmm-gui.org/)
- `CGUSER`: Your CHARMM-GUI email/username
- `CGPASS`: Your CHARMM-GUI password
- `BROWSER_TYPE`: `firefox` or `chrome`

## 2. Installation

Install the following dependencies:

### Python
- Python 3.8 or any version compatible with the multiprocessing requirements of Selenium/Splinter

### Python Libraries
Install these with `pip`:

- selenium
- splinter
- pyyaml
- mdtraj
- requests
- pdbfixer
- openmm
- yaml
- llama_index
- llama-index-llms-google-genai
- llama-index-embeddings-google-genai

Example installation command:
```bash
pip install selenium splinter pyyaml mdtraj requests pdbfixer openmm PyYAML llama_index llama-index-llms-google-genai llama-index-embeddings-google-genai
```

### System Requirements
- [geckodriver](https://github.com/mozilla/geckodriver/releases) (required if using Firefox)
  - Download and add to your PATH.

## 3. Running the Agentic Pipeline

The main entry point for the agentic pipeline is the Jupyter notebook:

```
new_rag/namd_agent_master.ipynb
```

Open this notebook in Jupyter and follow the instructions in the cells to:
- Generate and clean YAML configuration files for CHARMM-GUI jobs
- Run CHARMM-GUI workflows (solution or bilayer)
- Launch and monitor NAMD simulations
- Perform post-processing and analysis
- Use RAG (Retrieval-Augmented Generation) with LlamaIndex and Google Gemini for enhanced automation

**Note:**
- The notebook expects the Google Gemini API key to be set in the environment variable `GOOGLE_API_KEY` for LlamaIndex integration.
- Ensure all dependencies are installed and the config file is updated before running the notebook.

## 4. Additional Notes
- For advanced usage or troubleshooting, refer to the code and comments in `namd_agent_master.ipynb`.
- For legacy CHARMM-GUI automation and test case development, see the `auto_cgui_master/README.md`.

---

For questions or issues, please open an issue or contact the project maintainer. 