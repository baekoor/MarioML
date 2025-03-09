import subprocess
import sys
import os
import shutil
from pathlib import Path


def run_command(command, description=None):
    """Run a command and print its output"""
    if description:
        print(f"\n{description}...")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )

        # Print output in real-time
        for line in process.stdout:
            print(line.strip())

        process.wait()

        if process.returncode != 0:
            print(f"Warning: Command exited with code {process.returncode}")
            for line in process.stderr:
                print(f"Error: {line.strip()}")
        else:
            print("Command completed successfully")

        return process.returncode == 0
    except Exception as e:
        print(f"Error executing command: {e}")
        return False


def uninstall_global_packages():
    """Uninstall all the ML-related packages from global Python"""
    packages_to_uninstall = [
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "psutil",
        "tensorboard",
        "gym",
        "gym-super-mario-bros",
        "nes-py"
    ]

    print("\n===== UNINSTALLING GLOBAL PACKAGES =====")
    for package in packages_to_uninstall:
        run_command(
            f"{sys.executable} -m pip uninstall -y {package}",
            f"Uninstalling {package}"
        )


def create_venv(venv_path=".venv"):
    """Create a new virtual environment"""
    print("\n===== CREATING VIRTUAL ENVIRONMENT =====")

    # Check if venv already exists
    if os.path.exists(venv_path):
        choice = input(
            f"Virtual environment at '{venv_path}' already exists. Delete and recreate? (y/n): ")
        if choice.lower() == 'y':
            print(f"Removing existing virtual environment at '{venv_path}'")
            try:
                shutil.rmtree(venv_path)
            except Exception as e:
                print(f"Error removing existing venv: {e}")
                return False
        else:
            print("Using existing virtual environment")
            return True

    # Create new venv
    return run_command(
        f"{sys.executable} -m venv {venv_path}",
        "Creating new virtual environment"
    )


def install_in_venv(venv_path=".venv"):
    """Install packages inside the virtual environment"""
    print("\n===== INSTALLING PACKAGES IN VIRTUAL ENVIRONMENT =====")

    # Get the correct pip path
    if sys.platform == 'win32':
        pip_path = os.path.join(venv_path, "Scripts", "pip")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")

    # Install PyTorch with CUDA
    run_command(
        f"{pip_path} install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116",
        "Installing PyTorch with CUDA support"
    )

    # Install other dependencies
    dependencies = [
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "psutil==5.9.5",
        "tensorboard==2.11.2",
        "opencv-python",
        "gym==0.23.1",
        "nes-py==8.2.1",
        "gym-super-mario-bros==7.4.0"
    ]

    for dep in dependencies:
        run_command(
            f"{pip_path} install {dep}",
            f"Installing {dep.split('==')[0]}"
        )


def create_activation_scripts(venv_path=".venv"):
    """Create activation scripts for convenience"""
    print("\n===== CREATING ACTIVATION SCRIPTS =====")

    # Create batch file for Windows
    with open("activate_venv.bat", "w") as f:
        f.write(f"@echo off\n")
        f.write(f"echo Activating virtual environment at {venv_path}\n")
        f.write(f"call {venv_path}\\Scripts\\activate.bat\n")
        f.write(f"echo Environment activated! You can now run:\n")
        f.write(f"echo python mario_dqn.py\n")

    print("Created activate_venv.bat for Windows")

    # Create run mario demo batch file
    with open("run_mario.bat", "w") as f:
        f.write("@echo off\n")
        f.write("echo Running Mario Agent Demo\n")
        f.write("echo =======================\n\n")
        f.write("REM Activate virtual environment\n")
        f.write("call .venv\\Scripts\\activate\n\n")
        f.write("REM Check if virtual environment activated successfully\n")
        f.write("if errorlevel 1 (\n")
        f.write("    echo Failed to activate virtual environment.\n")
        f.write("    echo Make sure you have created the virtual environment.\n")
        f.write("    exit /b 1\n")
        f.write(")\n\n")
        f.write("REM Find latest checkpoint directory\n")
        f.write(
            "for /f \"tokens=*\" %%a in ('dir /b /od /ad .\\mario_checkpoints') do set LATEST_DIR=%%a\n\n")
        f.write("REM Find latest checkpoint file in that directory\n")
        f.write("for /f \"tokens=*\" %%a in ('dir /b /od .\\mario_checkpoints\\%LATEST_DIR%\\mario_model_step_*.pt') do set CHECKPOINT=.\\mario_checkpoints\\%LATEST_DIR%\\%%a\n\n")
        f.write("if \"%CHECKPOINT%\"==\"\" (\n")
        f.write("    echo No checkpoints found.\n")
        f.write("    exit /b 1\n")
        f.write(")\n\n")
        f.write("echo Latest checkpoint: %CHECKPOINT%\n")
        f.write("echo Running agent with latest checkpoint...\n\n")
        f.write("REM Run the agent with the found checkpoint\n")
        f.write(
            "python run_mario_agent.py --checkpoint \"%CHECKPOINT%\" --episodes 5\n\n")
        f.write("echo Demo complete!\n")
        f.write("pause\n")

    print("Created run_mario.bat for running demos")

    # Create verification script
    verification_script = """
import torch
import gym_super_mario_bros
import sys

print(f"Python: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available - check your PyTorch installation")

try:
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    print("Mario environment created successfully")
    env.close()
except Exception as e:
    print(f"Error creating Mario environment: {e}")
"""

    # Save verification script
    with open("verify_setup.py", "w") as f:
        f.write(verification_script)

    print("Created verify_setup.py to check installation")


if __name__ == "__main__":
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    choice = input(
        "This will uninstall ML packages from your global Python installation. Continue? (y/n): ")
    if choice.lower() != 'y':
        print("Operation cancelled")
        sys.exit(0)

    # Uninstall global packages commented out
    # uninstall_global_packages()

    # Create virtual environment
    venv_created = create_venv()

    if venv_created:
        # Install packages in the venv
        install_in_venv()

        # Create activation scripts
        create_activation_scripts()

        print("\n===== SETUP COMPLETE =====")
        print("To use the virtual environment:")
        print("1. Run 'activate_venv.bat' (Windows) to activate the environment")
        print("2. Run 'python verify_setup.py' to verify installation")
        print("3. Run 'python mario_dqn.py' to start training")
        print("4. After training, run 'run_mario.bat' to see your agent play")
    else:
        print("\nFailed to create virtual environment. Please check for errors.")
