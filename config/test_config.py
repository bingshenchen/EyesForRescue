# config/test_config.py

"""
Test script for the new configuration system.
Run this to verify that the migration was successful.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from config.settings import get_settings


    def test_configuration():
        """Test the new configuration system."""
        print("🧪 Testing EyesForRescue configuration...")
        print("=" * 50)

        try:
            # Get configuration
            config = get_settings()

            # Print configuration summary
            config.print_config_summary()

            # Test model paths
            print("\n🔍 Checking model files...")
            model_status = config.validate_models()

            if model_status:
                print("✅ All required models found!")
            else:
                print("⚠️  Some models are missing. This is expected after migration.")
                print("   You may need to move model files to the new locations.")

            # Create directories
            print("\n📁 Creating/verifying directories...")
            config.create_directories()

            # Test specific settings
            print("\n⚙️  Testing specific settings...")
            print(f"  Classes: {config.CLASSES}")
            print(f"  Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
            print(f"  Cache Enabled: {config.CACHE_ENABLED}")
            print(f"  GPU Enabled: {config.USE_GPU}")

            # Test paths
            print("\n📂 Testing key paths...")
            key_paths = {
                "Project Root": config.PROJECT_ROOT,
                "Data Directory": config.DATA_DIR,
                "Output Directory": config.OUTPUT_DIR,
                "YOLO Model": config.YOLO_MODEL_PATH,
                "Classifier": config.CLASSIFIER_PATH,
            }

            for name, path in key_paths.items():
                exists = "✅" if path.exists() else "❌"
                print(f"  {exists} {name}: {path}")

            print("\n" + "=" * 50)
            print("✅ Configuration test completed successfully!")
            print("\n📝 Next steps:")
            print("1. Move model files to new locations if needed")
            print("2. Update import statements in existing code")
            print("3. Test core functionality")

        except Exception as e:
            print(f"❌ Configuration test failed: {str(e)}")
            print("\n🔧 Troubleshooting tips:")
            print("1. Make sure config/settings.py exists")
            print("2. Check that .env file is updated")
            print("3. Verify PROJECT_ROOT path in .env")


    def check_migration_status():
        """Check the status of the migration."""
        print("📊 Migration Status Check")
        print("=" * 30)

        # Check if new directories exist
        new_dirs = [
            "config",
            "data",
            "data/models",
            "data/cache",
            "outputs",
            "src/core"
        ]

        for dir_name in new_dirs:
            dir_path = project_root / dir_name
            status = "✅" if dir_path.exists() else "❌"
            print(f"{status} {dir_name}")

        # Check if migration summary exists
        summary_path = project_root / "MIGRATION_SUMMARY.md"
        if summary_path.exists():
            print("✅ Migration summary created")
        else:
            print("❌ Migration summary not found")


    if __name__ == "__main__":
        print("🔄 EyesForRescue Configuration Test")
        print("=" * 40)

        # First check migration status
        check_migration_status()
        print()

        # Then test configuration
        test_configuration()

except ImportError as e:
    print("❌ Cannot import configuration module!")
    print(f"Error: {e}")
    print("\n🔧 Please ensure:")
    print("1. config/settings.py file exists")
    print("2. config/__init__.py file exists")
    print("3. .env file is properly configured")
    print("\nCurrent working directory:", os.getcwd())
    print("Project root:", project_root)