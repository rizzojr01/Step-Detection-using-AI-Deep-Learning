#!/usr/bin/env python3
"""
Project launcher for Step Detection System.
Provides easy access to all project functionalities.
"""

import os
import subprocess
import sys


def run_notebook():
    """Launch Jupyter notebook with the clean training notebook."""
    print("🔬 Launching Jupyter Notebook...")
    try:
        subprocess.run(
            ["jupyter", "notebook", "notebooks/CNN_TensorFlow_Clean.ipynb"], check=True
        )
    except subprocess.CalledProcessError:
        print("❌ Error launching Jupyter. Make sure it's installed:")
        print("pip install jupyter")
    except FileNotFoundError:
        print("❌ Jupyter not found. Install with:")
        print("pip install jupyter")


def run_main():
    """Run the main CLI interface."""
    print("🚀 Starting Main CLI Interface...")
    subprocess.run([sys.executable, "main.py"])


def run_tests():
    """Run the test suite."""
    print("🧪 Running Tests...")
    try:
        subprocess.run(["pytest", "tests/", "-v"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Some tests failed.")
    except FileNotFoundError:
        print("❌ pytest not found. Install with:")
        print("pip install pytest")


def install_package():
    """Install the package in development mode."""
    print("📦 Installing package in development mode...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Package installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing package.")


def install_requirements():
    """Install all requirements."""
    print("📋 Installing requirements...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing requirements.")


def show_structure():
    """Show project structure."""
    print("📁 Project Structure:")
    try:
        subprocess.run(["tree", "-I", "__pycache__|*.pyc|.git|.venv|.DS_Store", "-a"])
    except FileNotFoundError:
        print("tree command not available. Here's the basic structure:")
        for root, dirs, files in os.walk("."):
            level = root.replace(".", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Show only first 5 files per directory
                if not file.startswith(".") and not file.endswith(".pyc"):
                    print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")


def main():
    """Main launcher menu."""
    print("🚶‍♂️ Step Detection Project Launcher")
    print("=" * 50)
    print("Choose an option:")
    print("1. 📓 Open Jupyter Notebook (training)")
    print("2. 🚀 Run Main CLI Interface")
    print("3. 🧪 Run Tests")
    print("4. 📦 Install Package (development mode)")
    print("5. 📋 Install Requirements")
    print("6. 📁 Show Project Structure")
    print("7. ❓ Help")
    print("8. 🚪 Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-8): ").strip()

            if choice == "1":
                run_notebook()
            elif choice == "2":
                run_main()
            elif choice == "3":
                run_tests()
            elif choice == "4":
                install_package()
            elif choice == "5":
                install_requirements()
            elif choice == "6":
                show_structure()
            elif choice == "7":
                print("\n📚 Help:")
                print("• Option 1: Opens the clean Jupyter notebook for model training")
                print(
                    "• Option 2: Runs the main CLI with training, testing, and API options"
                )
                print("• Option 3: Runs all unit tests")
                print(
                    "• Option 4: Installs the step_detection package in development mode"
                )
                print("• Option 5: Installs all required dependencies")
                print("• Option 6: Shows the organized project structure")
                print("\n🔍 Project Overview:")
                print("This is a complete step detection system using TensorFlow CNNs.")
                print(
                    "The project is organized into packages for easy development and deployment."
                )
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-8.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
