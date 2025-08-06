#!/bin/bash
# MAC_Bench One-Click Setup Script
# This script installs all dependencies and downloads the dataset

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}🚀 MAC_Bench Setup Script${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_status "Please install Anaconda or Miniconda first:"
        print_status "https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda found: $(conda --version)"
}

# Create or update conda environment
setup_environment() {
    print_status "Setting up conda environment from environment.yml..."
    
    # Check if environment already exists
    if conda env list | grep -q "MAC_Bench"; then
        print_warning "MAC_Bench environment already exists"
        read -p "Do you want to update it? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Updating existing environment..."
            conda env update -f environment.yml
        else
            print_status "Skipping environment update"
        fi
    else
        print_status "Creating new environment..."
        conda env create -f environment.yml
    fi
    
    print_success "Conda environment setup complete"
}

# Install CLI dependencies
install_cli_dependencies() {
    print_status "Installing CLI dependencies..."
    
    # Activate the environment and install CLI requirements
    eval "$(conda shell.bash hook)"
    conda activate MAC_Bench
    
    print_status "Installing CLI-specific packages..."
    pip install -r requirements-cli.txt
    
    # Install additional dependencies for dataset download and CLI functionality
    print_status "Installing additional dependencies..."
    pip install huggingface_hub datasets curl_cffi
    
    print_success "CLI dependencies installed"
}

# Download dataset from Hugging Face
download_dataset() {
    print_status "Downloading dataset from Hugging Face..."
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate MAC_Bench
    
    # Run Python download script
    print_status "Running dataset download script..."
    python download_dataset.py
    
    if [ $? -eq 0 ]; then
        print_success "Dataset download completed"
    else
        print_error "Dataset download failed"
        print_status "Please check your internet connection and Hugging Face Hub access"
        return 1
    fi
}

# Setup CLI
setup_cli() {
    print_status "Setting up CLI..."
    
    # Make CLI script executable
    chmod +x mac
    
    # Test CLI installation
    print_status "Testing CLI installation..."
    eval "$(conda shell.bash hook)"
    conda activate MAC_Bench
    
    if ./mac --help > /dev/null 2>&1; then
        print_success "CLI setup complete and working"
    else
        print_error "CLI setup failed"
        return 1
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    eval "$(conda shell.bash hook)"
    conda activate MAC_Bench
    
    # Check CLI status
    print_status "Running system status check..."
    ./mac status
    
    # Check dataset
    if [ -d "MAC_Bench" ]; then
        dataset_size=$(du -sh MAC_Bench | cut -f1)
        print_success "Dataset available: $dataset_size"
    else
        print_warning "Dataset not found"
    fi
}

# Main installation process
main() {
    print_header
    
    print_status "Starting MAC_Bench installation..."
    print_status "This will:"
    print_status "  1. Check conda installation"
    print_status "  2. Create/update conda environment from environment.yml"
    print_status "  3. Install CLI dependencies"
    print_status "  4. Download dataset from Hugging Face"
    print_status "  5. Setup and verify CLI"
    print_status ""
    
    read -p "Continue with installation? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_status "Installation cancelled"
        exit 0
    fi
    
    # Run installation steps
    check_conda
    setup_environment
    install_cli_dependencies
    download_dataset
    setup_cli
    verify_installation
    
    print_header
    print_success "🎉 MAC_Bench installation completed!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Activate environment: conda activate MAC_Bench"
    print_status "2. Check system status: ./mac status"
    print_status "3. Create config: ./mac config template --output config.yaml"
    print_status "4. Run experiment: ./mac run --config config.yaml"
    print_status ""
    print_status "📖 For detailed usage instructions, see README.md"
}

# Handle script interruption
trap 'print_error "Installation interrupted"; exit 1' INT TERM

# Run main function
main "$@"