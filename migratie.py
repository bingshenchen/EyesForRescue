"""
Project structure migration script for EyesForRescue.
Organizes files into the new clean structure while preserving functionality.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List


class ProjectMigrator:
    """Handles migration of project files to new structure."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_created = False

    def create_new_structure(self):
        """Create the new directory structure."""
        new_dirs = [
            # Config directory
            'config',

            # Data directory structure
            'data',
            'data/datasets',
            'data/models',
            'data/models/yolo',
            'data/models/classifier',
            'data/cache',
            'data/cache/detections',

            # Output directory structure
            'outputs',
            'outputs/training_runs',
            'outputs/evaluation_results',
            'outputs/reports',
            'outputs/processed_videos',

            # Source code reorganization
            'src/core',
            'src/core/detection',
            'src/core/tracking',
            'src/core/analysis',
            'src/core/utils',

            # Scripts directory
            'scripts',
        ]

        for dir_path in new_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")

    def migrate_models(self):
        """Move model files to the new models directory."""
        model_migrations = {
            # YOLO models
            'src/train/models/best1.4.pt': 'data/models/yolo/best.pt',
            'src/train/models/yolo11n-pose.pt': 'data/models/yolo/yolo11n-pose.pt',

            # Classifier models
            'assets/classifier/classifier.pkl': 'data/models/classifier/rf_classifier.pkl',
            'src/train/classifier_test/final_person_help_classifier.keras': 'data/models/classifier/cnn_classifier.keras',

            # Additional classifier files
            'assets/classifier/fine_features.pkl': 'data/models/classifier/fine_features.pkl',
            'assets/classifier/fine_labels.pkl': 'data/models/classifier/fine_labels.pkl',
            'assets/classifier/needhelp_feature.pkl': 'data/models/classifier/needhelp_features.pkl',
            'assets/classifier/needhelp_labels.pkl': 'data/models/classifier/needhelp_labels.pkl',
        }

        self._migrate_files(model_migrations, "models")

    def migrate_core_modules(self):
        """Reorganize core source code modules."""
        core_migrations = {
            # Detection modules
            'src/main_app/demoForKlant/ground_truh_pro_yolo_rf_classifier.py': 'src/core/detection/fall_detector.py',
            'src/main_app/utils/detection/fall_detection.py': 'src/core/detection/yolo_detector.py',
            'src/main_app/poging/classifier.py': 'src/core/detection/pose_classifier.py',

            # Tracking modules
            'src/main_app/poging/tracker.py': 'src/core/tracking/sort_tracker.py',

            # Analysis modules
            'src/main_app/demoForKlant/calculate_danger_ad.py': 'src/core/analysis/danger_calculator.py',
            'src/main_app/demoForKlant/poging_gen.py': 'src/core/analysis/gpt_analyzer.py',
            'src/main_app/demoForKlant/gps.py': 'src/core/analysis/location_service.py',
            'src/main_app/demoForKlant/getweer.py': 'src/core/analysis/weather_service.py',

            # Utility modules
            'src/main_app/utils/cache_manager.py': 'src/core/utils/cache_manager.py',
            'src/main_app/utils/performance_analyzer.py': 'src/core/utils/performance_analyzer.py',
            'src/main_app/utils/video_processing.py': 'src/core/utils/video_processor.py',
        }

        self._migrate_files(core_migrations, "core modules")

    def create_config_files(self):
        """Create configuration files in the new structure."""
        # Create __init__.py files
        init_files = [
            'config/__init__.py',
            'src/core/__init__.py',
            'src/core/detection/__init__.py',
            'src/core/tracking/__init__.py',
            'src/core/analysis/__init__.py',
            'src/core/utils/__init__.py',
        ]

        for init_file in init_files:
            init_path = self.project_root / init_file
            if not init_path.exists():
                init_path.write_text('"""Package initialization."""\n')
                print(f"✓ Created: {init_file}")

    def create_migration_summary(self):
        """Create a summary of what was migrated."""
        summary_path = self.project_root / 'MIGRATION_SUMMARY.md'

        summary_content = """# Migration Summary

## Files Moved

### Models
- `src/train/models/best1.4.pt` → `data/models/yolo/best.pt`
- `src/train/models/yolo11n-pose.pt` → `data/models/yolo/yolo11n-pose.pt`
- `assets/classifier/classifier.pkl` → `data/models/classifier/rf_classifier.pkl`

### Core Modules
- `src/main_app/demoForKlant/ground_truh_pro_yolo_rf_classifier.py` → `src/core/detection/fall_detector.py`
- `src/main_app/demoForKlant/calculate_danger_ad.py` → `src/core/analysis/danger_calculator.py`
- `src/main_app/utils/cache_manager.py` → `src/core/utils/cache_manager.py`

## New Structure Created
- `config/` - Configuration management
- `data/` - All data, models, and cache
- `outputs/` - All output files
- `src/core/` - Reorganized core modules

## Next Steps
1. Update import statements in remaining files
2. Test all functionality with new structure
3. Update documentation and README
"""

        summary_path.write_text(summary_content)
        print(f"✓ Created migration summary: MIGRATION_SUMMARY.md")

    def _migrate_files(self, file_map: Dict[str, str], category: str):
        """Helper method to migrate files according to a mapping."""
        print(f"\n📁 Migrating {category}...")

        for source, destination in file_map.items():
            source_path = self.project_root / source
            dest_path = self.project_root / destination

            if source_path.exists():
                # Create destination directory if it doesn't exist
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file (keep original for safety)
                shutil.copy2(source_path, dest_path)
                print(f"  ✓ {source} → {destination}")
            else:
                print(f"  ⚠️  Source not found: {source}")

    def update_env_file(self):
        """Update .env file with new paths."""
        env_path = self.project_root / '.env'

        # Read current .env
        if env_path.exists():
            # Create backup
            shutil.copy2(env_path, self.project_root / '.env.backup')
            print("✓ Created .env backup")

        # The new .env content should be manually updated using the provided config
        print("⚠️  Please update your .env file with the new configuration provided earlier")

    def run_migration(self):
        """Run the complete migration process."""
        print("🔄 Starting project structure migration...")
        print(f"📁 Project root: {self.project_root}")

        try:
            # Step 1: Create new directory structure
            print("\n📂 Creating new directory structure...")
            self.create_new_structure()

            # Step 2: Migrate model files
            print("\n🤖 Migrating model files...")
            self.migrate_models()

            # Step 3: Migrate core modules
            print("\n🔧 Migrating core modules...")
            self.migrate_core_modules()

            # Step 4: Create configuration files
            print("\n⚙️  Creating configuration files...")
            self.create_config_files()

            # Step 5: Update environment variables
            print("\n📝 Updating environment configuration...")
            self.update_env_file()

            # Step 6: Create migration summary
            print("\n📋 Creating migration summary...")
            self.create_migration_summary()

            print("\n✅ Migration completed successfully!")
            print("\n📋 Next steps:")
            print("1. Update your .env file with the new configuration")
            print("2. Test the application with the new structure")
            print("3. Update import statements if needed")
            print("4. Remove old files after confirming everything works")

        except Exception as e:
            print(f"\n❌ Migration failed: {str(e)}")
            raise


def main():
    """Main migration function."""
    # Get project root from environment or current directory
    project_root = os.getenv('PROJECT_ROOT')
    if not project_root:
        project_root = input("Enter your project root path: ").strip('"')

    if not project_root:
        print("❌ Project root path is required!")
        return

    migrator = ProjectMigrator(project_root)

    # Confirm before proceeding
    response = input(f"\n🔄 Migrate project structure in {project_root}? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        migrator.run_migration()
    else:
        print("❌ Migration cancelled.")


if __name__ == "__main__":
    main()