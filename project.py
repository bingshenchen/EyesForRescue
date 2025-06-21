"""
Project Structure Printer for EyesForRescue
Prints a clean tree view of the project directory structure.
"""

import os
from pathlib import Path
from typing import Set, List


class ProjectStructurePrinter:
    """Prints project directory structure in a clean tree format."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

        # Files and directories to ignore
        self.ignore_patterns = {
            # Python cache and temp files
            '__pycache__',
            '*.pyc',
            '.pytest_cache',

            # Virtual environments
            'venv',
            'env',
            '.venv',
            '.env',

            # IDE files
            '.idea',
            '.vscode',
            '*.swp',
            '*.swo',

            # Git
            '.git',

            # OS files
            '.DS_Store',
            'Thumbs.db',

            # Large data directories (optional)
            'datasets',  # Comment this out if you want to see datasets
            'cache',

            # Log files
            '*.log',

            # Backup files
            '*.backup',
            '*.bak',
        }

        # File extensions to highlight
        self.important_extensions = {
            '.py': '🐍',
            '.yaml': '⚙️',
            '.yml': '⚙️',
            '.json': '📋',
            '.md': '📝',
            '.txt': '📄',
            '.pt': '🤖',
            '.pkl': '💾',
            '.keras': '🧠',
            '.env': '🔧',
        }

    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        name = path.name

        # Check exact matches
        if name in self.ignore_patterns:
            return True

        # Check pattern matches
        for pattern in self.ignore_patterns:
            if '*' in pattern:
                if pattern.startswith('*'):
                    if name.endswith(pattern[1:]):
                        return True
                elif pattern.endswith('*'):
                    if name.startswith(pattern[:-1]):
                        return True

        return False

    def get_file_icon(self, file_path: Path) -> str:
        """Get icon for file based on extension."""
        suffix = file_path.suffix.lower()
        return self.important_extensions.get(suffix, '📎')

    def print_tree(self, directory: Path = None, prefix: str = "", max_depth: int = 5, current_depth: int = 0):
        """Print directory tree structure."""
        if directory is None:
            directory = self.project_root

        if current_depth > max_depth:
            return

        if self.should_ignore(directory):
            return

        # Get all items in directory
        try:
            items = list(directory.iterdir())
        except PermissionError:
            return

        # Filter out ignored items
        items = [item for item in items if not self.should_ignore(item)]

        # Sort: directories first, then files
        items.sort(key=lambda x: (x.is_file(), x.name.lower()))

        for i, item in enumerate(items):
            is_last = i == len(items) - 1

            # Choose the right tree symbols
            current_prefix = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if item.is_dir():
                print(f"{prefix}{current_prefix}📁 {item.name}/")
                self.print_tree(item, next_prefix, max_depth, current_depth + 1)
            else:
                icon = self.get_file_icon(item)
                print(f"{prefix}{current_prefix}{icon} {item.name}")

    def print_summary(self):
        """Print a summary of the project structure."""
        print("=" * 60)
        print("📊 Project Structure Summary")
        print("=" * 60)

        # Count different types of files
        file_counts = {}
        dir_count = 0

        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)

            # Skip ignored directories
            if self.should_ignore(root_path):
                continue

            # Count directories
            for d in dirs:
                if not self.should_ignore(Path(root) / d):
                    dir_count += 1

            # Count files by extension
            for f in files:
                file_path = Path(root) / f
                if not self.should_ignore(file_path):
                    ext = file_path.suffix.lower() or 'no_extension'
                    file_counts[ext] = file_counts.get(ext, 0) + 1

        print(f"📁 Total Directories: {dir_count}")
        print(f"📄 Total Files: {sum(file_counts.values())}")
        print("\n📋 File Types:")

        # Sort by count (descending)
        sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_files[:10]:  # Top 10 file types
            icon = self.important_extensions.get(ext, '📎')
            ext_name = ext if ext != 'no_extension' else '(no extension)'
            print(f"  {icon} {ext_name}: {count} files")

        print("=" * 60)

    def print_key_files(self):
        """Print locations of key project files."""
        print("\n🔍 Key Project Files:")
        print("-" * 30)

        key_files = [
            # Configuration files
            '.env',
            '.env.example',
            '.env.template',
            'requirements.txt',
            'README.md',

            # Main application files
            'src/main.py',
            'src/main_app/app.py',
            'config/settings.py',

            # Model files
            'src/train/models/best1.4.pt',
            'assets/classifier/classifier.pkl',

            # Dataset config
            'assets/datasets/fall_detection/dataset.yaml',
        ]

        for file_path in key_files:
            full_path = self.project_root / file_path
            status = "✅" if full_path.exists() else "❌"
            icon = self.get_file_icon(full_path)
            print(f"  {status} {icon} {file_path}")

    def generate_full_report(self, max_depth: int = 4):
        """Generate a complete project structure report."""
        print("🌳 EyesForRescue Project Structure")
        print("=" * 60)
        print(f"📁 Root: {self.project_root}")
        print(f"🔍 Max Depth: {max_depth}")
        print("=" * 60)

        # Print the tree structure
        print(f"\n📁 {self.project_root.name}/")
        self.print_tree(max_depth=max_depth)

        # Print summary statistics
        self.print_summary()

        # Print key files status
        self.print_key_files()

        print("\n" + "=" * 60)
        print("✅ Structure analysis completed!")


def main():
    """Main function to run the structure printer."""
    # Get project root
    project_root = os.getenv('PROJECT_ROOT')
    if not project_root:
        # Try to detect from current location
        current_dir = Path.cwd()
        if 'AI-Applications' in str(current_dir):
            # Find the AI-Applications directory
            parts = current_dir.parts
            ai_app_index = next(i for i, part in enumerate(parts) if 'AI-Applications' in part)
            project_root = Path(*parts[:ai_app_index+1])
        else:
            project_root = input("Enter your project root path: ").strip('"')

    if not project_root:
        print("❌ Project root path is required!")
        return

    # Create printer and generate report
    printer = ProjectStructurePrinter(project_root)

    print("🎯 Project Structure Analysis")
    print("Choose an option:")
    print("1. Full structure report (recommended)")
    print("2. Tree view only")
    print("3. Summary only")
    print("4. Key files only")

    choice = input("\nEnter choice (1-4) or press Enter for full report: ").strip()

    if choice == "2":
        print(f"\n📁 {Path(project_root).name}/")
        printer.print_tree()
    elif choice == "3":
        printer.print_summary()
    elif choice == "4":
        printer.print_key_files()
    else:
        # Default: full report
        printer.generate_full_report()


if __name__ == "__main__":
    main()