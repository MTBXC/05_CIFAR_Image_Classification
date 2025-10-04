#!/usr/bin/env python3
"""
Script to check system requirements for Docker and AWS deployment.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_command(command, name):
    """Check if a command is available."""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ {name}: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {name}: Not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"❌ {name}: Not found")
        return False

def check_file_exists(file_path, name):
    """Check if a file exists."""
    if Path(file_path).exists():
        print(f"✅ {name}: Found")
        return True
    else:
        print(f"❌ {name}: Not found")
        return False

def check_python_packages():
    """Check required Python packages."""
    required_packages = [
        'tensorflow', 'fastapi', 'uvicorn', 'numpy', 
        'pillow', 'boto3', 'docker'
    ]
    
    print("\n📦 Checking Python packages:")
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Not installed")
            all_installed = False
    
    return all_installed

def main():
    """Main check function."""
    print("🔍 Checking system requirements for CIFAR-10 API deployment...\n")
    
    # Check system commands
    print("🛠️  Checking system tools:")
    docker_ok = check_command('docker', 'Docker')
    aws_ok = check_command('aws', 'AWS CLI')
    
    # Check project files
    print("\n📁 Checking project files:")
    dockerfile_ok = check_file_exists('Dockerfile', 'Dockerfile')
    compose_ok = check_file_exists('docker-compose.yml', 'docker-compose.yml')
    model_ok = check_file_exists('models/base_cnn_cifar10_cpu.h5', 'Trained model')
    data_ok = check_file_exists('data/raw/cifar-10-batches-py', 'CIFAR-10 data')
    
    # Check Python packages
    packages_ok = check_python_packages()
    
    # Summary
    print("\n📋 Summary:")
    all_requirements = all([
        docker_ok, aws_ok, dockerfile_ok, compose_ok, 
        model_ok, data_ok, packages_ok
    ])
    
    if all_requirements:
        print("🎉 All requirements met! Ready for deployment.")
    else:
        print("⚠️  Some requirements are missing. See installation guide below.")
        print_installation_guide(docker_ok, aws_ok, packages_ok)

def print_installation_guide(docker_ok, aws_ok, packages_ok):
    """Print installation guide for missing components."""
    print("\n📖 Installation Guide:")
    
    if not docker_ok:
        print("\n🐳 Docker Installation:")
        print("1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop")
        print("2. Install and restart your computer")
        print("3. Start Docker Desktop")
        print("4. Verify with: docker --version")
    
    if not aws_ok:
        print("\n☁️  AWS CLI Installation:")
        print("1. Download AWS CLI from: https://aws.amazon.com/cli/")
        print("2. Or install via pip: pip install awscli")
        print("3. Configure with: aws configure")
        print("4. Verify with: aws --version")
    
    if not packages_ok:
        print("\n🐍 Python Packages Installation:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. For Docker support: pip install docker")
        print("3. For AWS support: pip install boto3")

if __name__ == '__main__':
    main()



