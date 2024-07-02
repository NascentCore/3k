#!/bin/bash

# Determine the operating system
OS=$(uname -s)

# Define the download URLs for each OS
URL_MACOS="https://sxwl-ai.oss-cn-beijing.aliyuncs.com/artifacts/tools/sxwlctl-darwin.tar.gz"
URL_LINUX="https://sxwl-ai.oss-cn-beijing.aliyuncs.com/artifacts/tools/sxwlctl-linux.tar.gz"

# Function to download and install sxwlctl
install_sxwlctl() {
    local url=$1
    local file_name=$2

    echo "Downloading sxwlctl..."
    curl -sL $url -o $file_name

    echo "Extracting sxwlctl..."
    tar -xzf $file_name

    echo "Installing sxwlctl..."
    sudo mv sxwlctl /usr/local/bin/

    echo "Cleaning up..."
    rm $file_name

    # Verify installation
    if command -v sxwlctl &> /dev/null; then
        echo "sxwlctl was installed successfully!"
    else
        echo "Installation failed. Please try again."
    fi
}

# Function to create the ~/.sxwlctl.yaml configuration file
create_config_file() {
    local config_file="$HOME/.sxwlctl.yaml"

    echo "Creating configuration file at $config_file..."
    cat <<EOL > $config_file
token: ""
auth_url: "https://llm.sxwl.ai/api/resource/uploader_access"
bucket: sxwl-cache
endpoint: https://oss-cn-beijing.aliyuncs.com
EOL

    echo "Configuration file created successfully!"
    echo "请修改配置文件~/.sxwlctl.yaml中token为您自己的算想云AccessToken"
}

# Determine the correct URL and file name based on the OS
case "$OS" in
    Darwin)
        URL=$URL_MACOS
        FILE_NAME="sxwlctl-darwin.tar.gz"
        ;;
    Linux)
        URL=$URL_LINUX
        FILE_NAME="sxwlctl-linux.tar.gz"
        ;;
    *)
        echo "Unsupported operating system: $OS"
        exit 1
        ;;
esac

# Install sxwlctl
install_sxwlctl $URL $FILE_NAME

# Create the configuration file
create_config_file
