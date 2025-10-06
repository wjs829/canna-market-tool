#!/usr/bin/env python3
"""
Cannabis Analytics Tool Launcher
Simple script to run the different components of the cannabis analytics tool
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🌿 Cannabis Market Analytics Tool")
    print("=" * 50)
    print()
    print("Choose an option:")
    print("1. Run Data Collection")
    print("2. Run Analytics Engine")
    print("3. Launch Interactive Dashboard")
    print("4. Open Jupyter Notebook")
    print("5. Run All (Collect Data + Analytics)")
    print("6. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            print("\n🔄 Running data collection...")
            run_data_collection()
            break
        
        elif choice == "2":
            print("\n📊 Running analytics engine...")
            run_analytics()
            break
        
        elif choice == "3":
            print("\n🚀 Launching dashboard...")
            launch_dashboard()
            break
        
        elif choice == "4":
            print("\n📓 Opening Jupyter notebook...")
            open_notebook()
            break
        
        elif choice == "5":
            print("\n🔄 Running complete analysis...")
            run_data_collection()
            run_analytics()
            break
        
        elif choice == "6":
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")

def get_python_command():
    """Get the correct Python command for this environment"""
    # Try to use the virtual environment Python if available
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python.absolute())
    return "python"

def run_data_collection():
    """Run the data collection module"""
    try:
        python_cmd = get_python_command()
        result = subprocess.run([python_cmd, "src/data_collector.py"], 
                              cwd=".", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data collection completed successfully!")
            print(result.stdout)
        else:
            print("❌ Data collection failed:")
            print(result.stderr)
    
    except Exception as e:
        print(f"❌ Error running data collection: {e}")

def run_analytics():
    """Run the analytics engine"""
    try:
        python_cmd = get_python_command()
        result = subprocess.run([python_cmd, "src/analytics_engine.py"], 
                              cwd=".", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Analytics completed successfully!")
            print(result.stdout)
        else:
            print("❌ Analytics failed:")
            print(result.stderr)
    
    except Exception as e:
        print(f"❌ Error running analytics: {e}")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        python_cmd = get_python_command()
        print("🌐 Starting Streamlit dashboard...")
        print("📱 Dashboard will open in your web browser")
        print("🛑 Press Ctrl+C to stop the dashboard")
        print()
        
        # Run streamlit
        subprocess.run([python_cmd, "-m", "streamlit", "run", "dashboard.py"], cwd=".")
    
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

def open_notebook():
    """Open the Jupyter notebook"""
    try:
        python_cmd = get_python_command()
        notebook_path = "notebooks/cannabis_analytics.ipynb"
        
        print("📓 Starting Jupyter...")
        print("📱 Jupyter will open in your web browser")
        print("🛑 Press Ctrl+C to stop Jupyter")
        print()
        
        # Try to open specific notebook
        subprocess.run([python_cmd, "-m", "jupyter", "lab", notebook_path], cwd=".")
    
    except KeyboardInterrupt:
        print("\n🛑 Jupyter stopped by user")
    except Exception as e:
        print(f"❌ Error opening notebook: {e}")
        print("💡 Make sure Jupyter is installed: pip install jupyter")

if __name__ == "__main__":
    main()