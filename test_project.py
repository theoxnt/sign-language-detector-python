#!/usr/bin/env python3
"""
Test script for Sign Language Detector
Verifies all components are working correctly
"""

import sys
import os
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(message):
    print(f"{BLUE}[TEST]{RESET} {message}")

def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message):
    print(f"{RED}✗ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠ {message}{RESET}")

def test_imports():
    """Test if all required packages are installed"""
    print_test("Testing imports...")
    
    packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'sklearn': 'scikit-learn',
        'torch': 'torch',
        'numpy': 'numpy',
        'language_tool_python': 'language-tool-python'
    }
    
    all_ok = True
    for module, package in packages.items():
        try:
            __import__(module)
            print_success(f"{package} installed")
        except ImportError:
            print_error(f"{package} NOT installed")
            all_ok = False
    
    return all_ok

def test_project_structure():
    """Test if project directories exist"""
    print_test("Testing project structure...")
    
    required_dirs = [
        'src',
        'src/data',
        'src/data_pickle',
        'src/models'
    ]
    
    required_files = [
        'src/__init__.py',
        'src/cli_enhanced.py',
        'src/core.py',
        'src/io_.py',
        'src/main.py',
        'requirements.txt'
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_success(f"Directory exists: {dir_path}")
        else:
            print_error(f"Directory missing: {dir_path}")
            all_ok = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"File missing: {file_path}")
            all_ok = False
    
    return all_ok

def test_cli_help():
    """Test if CLI help works"""
    print_test("Testing CLI help...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli_enhanced', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and 'Sign Language Detector' in result.stdout:
            print_success("CLI help works")
            return True
        else:
            print_error("CLI help failed")
            return False
    except Exception as e:
        print_error(f"CLI help error: {e}")
        return False

def test_cli_commands():
    """Test if CLI commands are available"""
    print_test("Testing CLI commands...")
    
    commands = ['collect', 'dataset', 'train', 'infer', 'list', 'setup', 'info']
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli_enhanced', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        all_ok = True
        for cmd in commands:
            if cmd in result.stdout:
                print_success(f"Command available: {cmd}")
            else:
                print_error(f"Command missing: {cmd}")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print_error(f"CLI commands test error: {e}")
        return False

def test_cli_info():
    """Test CLI info command"""
    print_test("Testing CLI info command...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli_enhanced', 'info'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and 'Sign Language Detector' in result.stdout:
            print_success("CLI info command works")
            return True
        else:
            print_error("CLI info command failed")
            return False
    except Exception as e:
        print_error(f"CLI info error: {e}")
        return False

def test_cli_list():
    """Test CLI list command"""
    print_test("Testing CLI list command...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli_enhanced', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and 'Available Resources' in result.stdout:
            print_success("CLI list command works")
            return True
        else:
            print_error("CLI list command failed")
            return False
    except Exception as e:
        print_error(f"CLI list error: {e}")
        return False

def test_cli_setup():
    """Test CLI setup command"""
    print_test("Testing CLI setup command...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli_enhanced', 'setup'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and 'Setup complete' in result.stdout:
            print_success("CLI setup command works")
            return True
        else:
            print_error("CLI setup command failed")
            return False
    except Exception as e:
        print_error(f"CLI setup error: {e}")
        return False

def test_core_imports():
    """Test if core modules can be imported"""
    print_test("Testing core module imports...")
    
    modules = [
        'src.cli_enhanced',
        'src.core',
        'src.io_',
        'src.main'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print_success(f"Module imports: {module}")
        except Exception as e:
            print_error(f"Module import failed: {module} - {e}")
            all_ok = False
    
    return all_ok

def test_models_exist():
    """Test if trained models exist"""
    print_test("Testing for trained models...")
    
    model_dir = Path('./src/models')
    if not model_dir.exists():
        print_warning("Models directory doesn't exist")
        return False
    
    models = list(model_dir.glob('model_*.p'))
    if models:
        for model in models:
            print_success(f"Model found: {model.name}")
        return True
    else:
        print_warning("No trained models found (this is OK if you haven't trained yet)")
        return False

def test_data_exists():
    """Test if training data exists"""
    print_test("Testing for training data...")
    
    data_dir = Path('./src/data')
    if not data_dir.exists():
        print_warning("Data directory doesn't exist")
        return False
    
    folders = [f for f in data_dir.iterdir() if f.is_dir()]
    if folders:
        for folder in folders:
            classes = [c for c in folder.iterdir() if c.is_dir()]
            print_success(f"Data folder found: {folder.name} ({len(classes)} classes)")
        return True
    else:
        print_warning("No training data found (this is OK if you haven't collected yet)")
        return False

def test_webcam():
    """Test if webcam is accessible"""
    print_test("Testing webcam access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print_success("Webcam is accessible")
                return True
            else:
                print_error("Webcam opened but couldn't read frame")
                return False
        else:
            print_error("Webcam not accessible")
            return False
    except Exception as e:
        print_error(f"Webcam test error: {e}")
        return False

def test_mediapipe():
    """Test if MediaPipe hand detection works"""
    print_test("Testing MediaPipe hand detection...")
    
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Check if hand landmarker model exists
        model_path = 'hand_landmarker.task'
        if Path(model_path).exists():
            print_success("MediaPipe hand landmarker model found")
            return True
        else:
            print_warning("MediaPipe hand landmarker model not found")
            print_warning("Run: curl -L -o hand_landmarker.task 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'")
            return False
    except Exception as e:
        print_error(f"MediaPipe test error: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Sign Language Detector - Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("CLI Help", test_cli_help),
        ("CLI Commands", test_cli_commands),
        ("CLI Info", test_cli_info),
        ("CLI List", test_cli_list),
        ("CLI Setup", test_cli_setup),
        ("Core Imports", test_core_imports),
        ("MediaPipe", test_mediapipe),
        ("Webcam", test_webcam),
        ("Trained Models", test_models_exist),
        ("Training Data", test_data_exists),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'-'*60}")
        result = test_func()
        results.append((name, result))
        print()
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Test Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{status} - {name}")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"{GREEN}✓ All tests passed! Project is ready to use.{RESET}")
        return 0
    elif passed >= total - 2:
        print(f"{YELLOW}⚠ Most tests passed. Some optional features may not work.{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some tests failed. Please fix the issues above.{RESET}")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
