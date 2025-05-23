import os
import subprocess
from datetime import datetime
from pathlib import Path
from config.config import settings

BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)

def backup_database():
    """Create a backup of the database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"backup_{timestamp}.sql"

    # Extract database connection details from URL
    # postgresql://user:password@host:port/dbname
    url_parts = settings.DATABASE_URL.replace("postgresql://", "").split("@")
    auth, host_db = url_parts
    user, password = auth.split(":")
    host, dbname = host_db.split("/")

    # Create backup using pg_dump
    try:
        subprocess.run([
            "pg_dump",
            f"--host={host}",
            f"--port=5439",  # Using the mapped port
            f"--username={user}",
            f"--dbname={dbname}",
            f"--file={backup_file}",
            "--format=custom",  # Custom format for better compression
            "--no-owner",  # Don't output commands to set ownership
            "--no-acl"     # Don't output ACL (grant/revoke commands)
        ], env={"PGPASSWORD": password}, check=True)

        print(f"Backup created successfully: {backup_file}")
        return backup_file
    except subprocess.CalledProcessError as e:
        print(f"Error creating backup: {e}")
        return None

def restore_database(backup_file: Path):
    """Restore database from a backup file."""
    if not backup_file.exists():
        print(f"Backup file not found: {backup_file}")
        return False

    # Extract database connection details from URL
    url_parts = settings.DATABASE_URL.replace("postgresql://", "").split("@")
    auth, host_db = url_parts
    user, password = auth.split(":")
    host, dbname = host_db.split("/")

    # Restore using pg_restore
    try:
        subprocess.run([
            "pg_restore",
            f"--host={host}",
            f"--port=5439",  # Using the mapped port
            f"--username={user}",
            f"--dbname={dbname}",
            "--clean",  # Clean (drop) database objects before recreating
            "--no-owner",  # Don't output commands to set ownership
            "--no-acl",    # Don't output ACL (grant/revoke commands)
            str(backup_file)
        ], env={"PGPASSWORD": password}, check=True)

        print(f"Database restored successfully from {backup_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error restoring database: {e}")
        return False

def list_backups():
    """List all available backups."""
    backups = sorted(BACKUP_DIR.glob("backup_*.sql"), reverse=True)
    if not backups:
        print("No backups found.")
        return

    print("\nAvailable backups:")
    for i, backup in enumerate(backups, 1):
        size_mb = backup.stat().st_size / (1024 * 1024)
        print(f"{i}. {backup.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database backup and restore utility")
    parser.add_argument("action", choices=["backup", "restore", "list"])
    parser.add_argument("--file", help="Backup file to restore from")

    args = parser.parse_args()

    if args.action == "backup":
        backup_database()
    elif args.action == "restore":
        if not args.file:
            print("Please specify a backup file to restore from.")
            list_backups()
        else:
            restore_database(Path(args.file))
    elif args.action == "list":
        list_backups()