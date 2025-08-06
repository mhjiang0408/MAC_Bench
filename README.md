# MAC_Bench

[![arXiv](https://img.shields.io/badge/arXiv-2501.01234-b31b1b.svg)](https://arxiv.org/abs/2501.XXXXX)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/mhjiang0408/MAC_Bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **MAC** - A Live Benchmark for Multimodal Large Language Models in Scientific Understanding

## 🌟 Features

- **Two Task Types**: Image-to-Text and Text-to-Image understanding
- **Advanced Methods**: CoVR (Cover Vision Reasoning) methodology with multiple variants
- **Multiple Models**: Support for GPT-4O, Qwen-VL, Step-1V, Gemini, and more
- **Easy CLI**: Simple `mac run` and `mac analyze` commands
- **Comprehensive Analysis**: Automatic report generation with visualizations
- **Scientific Focus**: Real scientific journal covers from Nature, Science, Cell, etc.

## 🚀 Quick Start

### One-Click Installation

```bash
git clone https://github.com/mhjiang0408/MAC_Bench.git
cd MAC_Bench
./setup.sh
```

The setup script automatically:
- ✅ Creates conda environment from `environment.yml`
- ✅ Installs CLI dependencies  
- ✅ Downloads dataset from Hugging Face
- ✅ Sets up and verifies CLI tools

### Three-Step Usage

**1. Create Configuration**
```bash
mac config template --output config.yaml --type example
# Edit config.yaml with your API keys
```

**2. Run Experiment**
```bash
mac run --config config.yaml
```

**3. Analyze Results**
```bash
mac analyze experiment/results/
```

## ⚙️ Configuration

Create your configuration file:

```bash
mac config template --output config.yaml --type basic
```

Example configuration:

```yaml
models:
  - name: gpt-4o
    api_base: https://api.openai.com/v1
    api_key: sk-your-api-key
    prompt_template: Config/prompt_template/4_choice_template.json
    # If you are testing text2image tasks, you need to set prompt_template to Config/prompt_template/4_choice_template_given_cover_story.json
    resume: false
    resume_path: None
    num_workers: 4

data:
  data_path: MAC_Bench/image2text_info.csv
  output_folder: ./experiment/results/
  scaling_factor: 1.0
  num_options: 4
  type: image2text
  # If you are testing text2image tasks, you need to set type to text2image
  random_seed: 42
```

## 🎯 CLI Commands

### Main Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `mac run` | Run experiments | `mac run --config config.yaml` |
| `mac analyze` | Analyze results | `mac analyze experiment/results/` |
| `mac status` | Check system status | `mac status --detailed` |
| `mac config` | Manage configurations | `mac config validate config.yaml` |

### Common Options

**For `mac run`:**
- `--config config.yaml` - Configuration file
- `--models gpt-4o` - Run specific model
- `--scaling-factor 0.01` - Use 1% of data for testing
- `--dry-run` - Preview without running
- `--verbose` - Detailed output

**For `mac analyze`:**
- `--output reports/` - Output directory
- `--format html` - Report format (json/csv/html/all)
- `--compare exp2.csv` - Compare experiments
- `--detailed` - Include detailed analysis

## 🔬 Example Workflows

### Quick Test Run
```bash
# Test with 1% of data
mac run --config config.yaml --scaling-factor 0.01 --verbose
```

### Full Experiment
```bash
# Run all models from config
mac run --config config.yaml

# Analyze with comprehensive reports
mac analyze experiment/results/ --output reports/ --format all
```

### Compare Models
```bash
# Run specific models
mac run --config config.yaml --models gpt-4o --models qwen-vl-max

# Compare results
mac analyze results1.csv --compare results2.csv --plot
```

## 📊 Understanding Tasks

### Image-to-Text Task
- **Input**: Scientific journal cover image
- **Question**: "Which of the following options best describe the cover image?"
- **Options**: 4 text descriptions (A, B, C, D)
- **Goal**: Select the most accurate description

### Text-to-Image Task  
- **Input**: Journal cover story text
- **Question**: "Which image best describes the cover story?"
- **Options**: 4 candidate images (A, B, C, D)
- **Goal**: Select the matching image

## 📁 Project Structure

```
MAC_Bench/
├── mac                          # CLI entry point
├── setup.sh                     # One-click installation
├── download_dataset.py          # Dataset download script
├── environment.yml              # Conda environment
├── requirements-cli.txt         # CLI dependencies
│
├── mac_cli/                     # CLI implementation
│   ├── commands/                # CLI commands
│   ├── core/                    # Core functionality  
│   └── utils/                   # Utilities
│
├── Config/                      # Configuration files
│   └── prompt_template/         # Prompt templates
│
├── Dataset/                     # Dataset construction scripts
├── experiment/                  # Experiment code
│   ├── method/                  # CoVR implementations
│   └── understanding/           # Task implementations
│
├── utils/                       # Core utilities
├── MAC_Bench/                   # Downloaded dataset
└── experiment/results/          # Experiment outputs
```

## 🐛 Troubleshooting

### System Check
```bash
mac status --detailed  # Check what's wrong
```

### Common Issues

**Environment Problems:**
```bash
conda env update -f environment.yml
```

**Missing Dependencies:**
```bash
pip install -r requirements-cli.txt
```

**Dataset Download Issues:**
```bash
python download_dataset.py  # Manual download
```

**API Connection Problems:**
```bash
mac status --check-apis --config config.yaml
# Check your API keys in config.yaml
```

## 📚 Dataset Information

The MAC_Bench dataset is available on [🤗 Hugging Face](https://huggingface.co/datasets/mac-bench/MAC-Bench) and contains:
- **Source Journals**: Nature, Science, Cell, ACS Central Science
- **Cover Images**: High-resolution scientific journal covers  
- **Cover Stories**: Corresponding textual descriptions
- **Task Variants**: Image2Text and Text2Image understanding
- **Size**: 10,000+ samples across multiple journals

### Download Dataset
The dataset is automatically downloaded during setup, but you can also download it manually:
```bash
# Via Hugging Face
from datasets import load_dataset
dataset = load_dataset("mhjiang0408/MAC_Bench")

# Or via download script
python download_dataset.py
```

## 💡 Advanced Usage

### Performance Optimization
```bash
# More workers for faster processing
mac run --config config.yaml --workers 8

# Resume interrupted experiments
mac run --config config.yaml --resume
```

### Custom Analysis
```bash
# Group results by journal
mac analyze results/ --group-by journal

# Generate only JSON reports
mac analyze results/ --format json --no-plot
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Citation

If you use MAC_Bench in your research, please cite our paper:

```bibtex
@article{mac_bench_2025,
  title={MAC: A Live Benchmark for Multimodal Large Language Models in Scientific Understanding},
  author={},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025},
  url={https://arxiv.org/abs/2501.XXXXX}
}
```

## 🆘 Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions for help

