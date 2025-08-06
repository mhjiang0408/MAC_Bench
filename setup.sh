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

# Let user choose between conda and uv
choose_package_manager() {
    echo -e "${BLUE}Choose your Python package manager:${NC}"
    echo "1) Conda (default)"
    echo "2) uv (faster)"
    read -p "Enter choice [1/2]: " -n 1 -r
    echo ""
    
    case $REPLY in
        2)
            PACKAGE_MANAGER="uv"
            print_status "Using uv as package manager"
            ;;
        *)
            PACKAGE_MANAGER="conda" 
            print_status "Using conda as package manager"
            ;;
    esac
}

# Check if package manager is installed
check_package_manager() {
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        if ! command -v uv &> /dev/null; then
            print_error "uv is not installed or not in PATH"
            print_status "Please install uv first:"
            print_status "curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        print_success "uv found: $(uv --version)"
    else
        if ! command -v conda &> /dev/null; then
            print_error "Conda is not installed or not in PATH"
            print_status "Please install Anaconda or Miniconda first:"
            print_status "https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
        print_success "Conda found: $(conda --version)"
    fi
}

# Create or update environment
setup_environment() {
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        print_status "Setting up uv virtual environment with Python 3.9..."
        
        if [ -d ".venv" ]; then
            print_warning ".venv environment already exists"
            read -p "Do you want to recreate it? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_status "Removing existing environment..."
                rm -rf .venv
                print_status "Creating new environment with Python 3.9..."
                uv venv --python 3.9
            else
                print_status "Keeping existing environment"
            fi
        else
            print_status "Creating new environment with Python 3.9..."
            uv venv --python 3.9
        fi
    else
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
    fi
    
    print_success "Environment setup complete"
}

# Install CLI dependencies
install_cli_dependencies() {
    print_status "Installing dependencies..."
    
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        # Activate uv environment
        source .venv/bin/activate
        
        # Install Python 3.9 if needed (uv will use system Python or install if specified)
        print_status "Installing Python dependencies from environment.yml with uv..."
        
        # Extract pip dependencies from environment.yml and install them
        if [ -f "environment.yml" ]; then
            print_status "Extracting dependencies from environment.yml..."
            # First install PyYAML to parse the file
            uv pip install PyYAML
            
            # Create temporary requirements file from environment.yml pip section
            python -c "
import yaml
with open('environment.yml', 'r') as f:
    env = yaml.safe_load(f)
    
pip_deps = []
for dep in env.get('dependencies', []):
    if isinstance(dep, dict) and 'pip' in dep:
        pip_deps.extend(dep['pip'])

with open('temp_requirements.txt', 'w') as f:
    for dep in pip_deps:
        f.write(dep + '\n')
"
            
            print_status "Installing dependencies with uv..."
            uv pip install -r temp_requirements.txt
            
            # Clean up temporary file
            rm -f temp_requirements.txt
        else
            print_warning "environment.yml not found, installing basic requirements only"
        fi
        
        print_status "Installing CLI-specific packages..."
        uv pip install -r requirements-cli.txt
        
        # Install additional dependencies for dataset download and CLI functionality
        print_status "Installing additional dependencies..."
        uv pip install huggingface_hub datasets curl_cffi
    else
        # Activate conda environment
        eval "$(conda shell.bash hook)"
        conda activate MAC_Bench
        
        print_status "Installing CLI-specific packages..."
        pip install -r requirements-cli.txt
        
        # Install additional dependencies for dataset download and CLI functionality
        print_status "Installing additional dependencies..."
        pip install huggingface_hub datasets curl_cffi
    fi
    
    print_success "Dependencies installed"
}

# Download dataset from Hugging Face
download_dataset() {
    print_status "Checking for existing dataset..."
    
    # Check if dataset already exists
    if [ -d "MAC_Bench" ] || [ -d "MAC_Bench_data" ] || [ -f "dataset_downloaded.flag" ]; then
        print_warning "Dataset appears to already exist"
        print_status "Found existing dataset files/directories"
        read -p "Do you want to re-download the dataset? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Skipping dataset download"
            return 0
        fi
        print_status "Proceeding with dataset download..."
    fi
    
    print_status "Downloading dataset from Hugging Face..."
    
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        # Activate uv environment
        source .venv/bin/activate
    else
        # Activate conda environment
        eval "$(conda shell.bash hook)"
        conda activate MAC_Bench
    fi
    
    # Run Python download script
    print_status "Running dataset download script..."
    python download_dataset.py
    
    if [ $? -eq 0 ]; then
        print_success "Dataset download completed"
        # Create a flag file to indicate successful download
        touch dataset_downloaded.flag
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
    
    # Ask user if they want to install mac to PATH
    echo ""
    print_status "Do you want to install 'mac' command to your PATH?"
    print_status "This will allow you to run 'mac --help' instead of './mac --help'"
    read -p "Install to PATH? [y/N]: " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Try different installation locations
        if [ -w "/usr/local/bin" ]; then
            INSTALL_DIR="/usr/local/bin"
        elif [ -w "$HOME/.local/bin" ]; then
            INSTALL_DIR="$HOME/.local/bin"
            # Add to PATH if not already there
            if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
                print_status "Adding $HOME/.local/bin to PATH..."
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
                export PATH="$HOME/.local/bin:$PATH"
            fi
        else
            # Create ~/.local/bin if it doesn't exist
            mkdir -p "$HOME/.local/bin"
            INSTALL_DIR="$HOME/.local/bin"
            if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
                print_status "Adding $HOME/.local/bin to PATH..."
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
                export PATH="$HOME/.local/bin:$PATH"
            fi
        fi
        
        print_status "Installing mac command to $INSTALL_DIR..."
        
        # Create a wrapper script that activates the environment first
        CURRENT_DIR=$(pwd)
        cat > "$INSTALL_DIR/mac" << EOF
#!/bin/bash
# MAC_Bench CLI wrapper script
PROJECT_DIR="$CURRENT_DIR"
cd "\$PROJECT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif command -v conda &> /dev/null && conda env list | grep -q "MAC_Bench"; then
    eval "\$(conda shell.bash hook)"
    conda activate MAC_Bench
fi

exec "\$PROJECT_DIR/mac" "\$@"
EOF
        
        chmod +x "$INSTALL_DIR/mac"
        print_success "Installed mac command to $INSTALL_DIR"
        print_status "You may need to restart your terminal or run 'source ~/.bashrc'"
    else
        print_status "Skipping PATH installation - use './mac' to run commands"
    fi
    
    echo ""
    
    # Test CLI installation
    print_status "Testing CLI installation..."
    
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        source .venv/bin/activate
    else
        eval "$(conda shell.bash hook)"
        conda activate MAC_Bench
    fi
    
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
    
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        source .venv/bin/activate
    else
        eval "$(conda shell.bash hook)"
        conda activate MAC_Bench
    fi
    
    # Check CLI status
    print_status "Running system status check..."
    ./mac status
    
    # Check dataset
    if [ -d "MAC_Bench" ] && [ "$PACKAGE_MANAGER" = "conda" ]; then
        dataset_size=$(du -sh MAC_Bench | cut -f1)
        print_success "Dataset available: $dataset_size"
    elif [ -d "MAC_Bench_data" ]; then
        dataset_size=$(du -sh MAC_Bench_data | cut -f1)
        print_success "Dataset available: $dataset_size"
    else
        print_warning "Dataset not found"
    fi
}

# Main installation process
main() {
    print_header
    
    # Choose package manager first
    choose_package_manager
    echo ""
    
    print_status "Starting MAC_Bench installation..."
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        print_status "This will:"
        print_status "  1. Check uv installation"
        print_status "  2. Create uv virtual environment"
        print_status "  3. Install CLI dependencies with uv"
        print_status "  4. Download dataset from Hugging Face"
        print_status "  5. Setup and verify CLI"
    else
        print_status "This will:"
        print_status "  1. Check conda installation"
        print_status "  2. Create/update conda environment from environment.yml"
        print_status "  3. Install CLI dependencies"
        print_status "  4. Download dataset from Hugging Face"
        print_status "  5. Setup and verify CLI"
    fi
    print_status ""
    
    read -p "Continue with installation? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_status "Installation cancelled"
        exit 0
    fi
    
    # Run installation steps
    check_package_manager
    setup_environment
    install_cli_dependencies
    download_dataset
    setup_cli
    verify_installation
    
    print_header
    print_success "🎉 MAC_Bench installation completed!"
    print_status ""
    print_status "Next steps:"
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        print_status "1. Activate environment: source .venv/bin/activate"
    else
        print_status "1. Activate environment: conda activate MAC_Bench"
    fi
    
    # Check if mac was installed to PATH
    if command -v mac &> /dev/null && [ "$(command -v mac)" != "./mac" ]; then
        print_status "2. Check system status: mac status"
        print_status "3. Create config: mac config template --output config.yaml"
        print_status "4. Run experiment: mac run --config config.yaml"
    else
        print_status "2. Check system status: ./mac status"
        print_status "3. Create config: ./mac config template --output config.yaml"
        print_status "4. Run experiment: ./mac run --config config.yaml"
    fi
    print_status ""
    print_status "📖 For detailed usage instructions, see README.md"
}

# Handle script interruption
trap 'print_error "Installation interrupted"; exit 1' INT TERM

# Run main function
main "$@"