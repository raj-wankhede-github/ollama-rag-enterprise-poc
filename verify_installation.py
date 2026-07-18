"""
Verification script to check all dependencies are correctly installed
and compatible with Python 3.13.8
"""

import sys
import importlib
from typing import Tuple, List
from importlib.metadata import version as get_version, PackageNotFoundError

# Expected versions - mapping package name to (expected_version, import_name)
EXPECTED_VERSIONS = {
    "fastapi": ("0.137.2", "fastapi"),
    "uvicorn": ("0.34.3", "uvicorn"),
    "pydantic": ("2.12.1", "pydantic"),
    "requests": ("2.32.4", "requests"),
    "chromadb": ("0.6.1", "chromadb"),
    "posthog": ("3.7.0", "posthog"),
    "numpy": ("2.1.3", "numpy"),
    "python-dotenv": ("1.0.1", "dotenv"),  # package name differs from import name
    "pypdf": ("5.0.1", "pypdf"),
    "pdfplumber": ("0.11.4", "pdfplumber"),
    "pymupdf": ("1.24.12", "fitz"),
    "Pillow": ("11.0.0", "PIL"),
    "pytesseract": ("0.3.13", "pytesseract"),
}


def check_python_version() -> bool:
    """Check if Python version is 3.13+"""
    version_info = sys.version_info
    print(f"Python Version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 13):
        print("[WARN] Python 3.13+ recommended for best compatibility")
        return False
    
    print("[OK] Python version is 3.13+")
    return True


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and get its version"""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        # First, try to import the module to verify it's installed
        module = importlib.import_module(import_name)
        
        # Try to get version from module attributes first
        version = (
            getattr(module, "__version__", None) or
            getattr(module, "version", None) or
            getattr(module, "VERSION", None)
        )
        
        # If no version in module, use importlib.metadata
        if not version:
            try:
                version = get_version(package_name)
            except PackageNotFoundError:
                version = "unknown"
        
        return True, str(version)
    except ImportError:
        return False, "not installed"


def verify_installation() -> int:
    """Verify all packages are installed correctly"""
    print("=" * 60)
    print("Dependency Verification for Ollama RAG Enterprise Demo")
    print("=" * 60)
    print()
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Check each package
    print("Checking installed packages:")
    print("-" * 60)
    
    all_ok = True
    for package_name, (expected_version, import_name) in EXPECTED_VERSIONS.items():
        installed, version = check_package(package_name, import_name)
        
        if installed:
            # Check if version matches expected
            version_match = str(version).startswith(expected_version)
            status = "[OK]" if version_match else "[WARN]"
            print(f"{status} {package_name:<20} {str(version):<15} (expected: {expected_version}*)")
            
            if not version_match:
                print(f"   Warning: Version may not be compatible")
                all_ok = False
        else:
            print(f"[FAIL] {package_name:<20} NOT INSTALLED")
            all_ok = False
    
    print()
    print("-" * 60)
    
    if all_ok and python_ok:
        print("[OK] All dependencies verified successfully!")
        print()
        print("Next steps:")
        print("1. Run: python main.py")
        print("2. Visit: http://localhost:8000/docs")
        return 0
    else:
        print("[FAIL] Some issues found. Please install missing packages:")
        print("   pip install -r requirements.txt")
        return 1


def verify_imports() -> int:
    """Verify all critical imports work"""
    print()
    print("=" * 60)
    print("Testing Critical Imports")
    print("=" * 60)
    print()
    
    imports_to_test = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "run"),
        ("pydantic", "BaseModel"),
        ("requests", "get"),
        ("chromadb", "Client"),
        ("posthog", "capture"),
        ("numpy", "array"),
        ("pypdf", "PdfReader"),
        ("pdfplumber", "open"),
        ("fitz", "open"),
        ("PIL", "Image"),
        ("pytesseract", "image_to_string"),
    ]
    
    all_ok = True
    for module_name, attr_name in imports_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, attr_name):
                print(f"[OK] {module_name}.{attr_name}")
            else:
                print(f"[WARN] {module_name} (missing {attr_name})")
                all_ok = False
        except ImportError as e:
            print(f"[FAIL] {module_name}: {e}")
            all_ok = False
    
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    result = verify_installation()
    verify_imports()
    sys.exit(result)
