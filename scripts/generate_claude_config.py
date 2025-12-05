#!/usr/bin/env python3
"""Generate Claude Desktop configuration snippet for this MCP server."""

import json
import os
import platform
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
venv_python = project_root / ".venv" / "bin" / "python"

# Determine Claude Desktop config location
system = platform.system()
if system == "Darwin":  # macOS
    config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
elif system == "Windows":
    config_path = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
else:  # Linux
    config_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

# Generate configuration
config = {
    "mcpServers": {
        "tabular-data": {
            "command": str(venv_python.absolute()),
            "args": [
                "-m",
                "mcp_tabular.server"
            ]
        }
    }
}

print("=" * 70)
print("Claude Desktop Configuration")
print("=" * 70)
print()
print(f"Config file location: {config_path}")
print()
print("Add this to your claude_desktop_config.json:")
print()
print(json.dumps(config, indent=2))
print()
print("=" * 70)
print()

# Check if config file exists
if config_path.exists():
    print(f"⚠️  Config file already exists at: {config_path}")
    print("   You'll need to merge this configuration manually.")
    print()
    
    # Try to read existing config
    try:
        with open(config_path, 'r') as f:
            existing = json.load(f)
        
        if "mcpServers" in existing:
            if "tabular-data" in existing["mcpServers"]:
                print("⚠️  'tabular-data' server already configured!")
            else:
                print("✅ Found existing MCP servers. Merge the configuration above.")
        else:
            print("✅ Config file exists but has no mcpServers. Add the config above.")
    except json.JSONDecodeError:
        print("⚠️  Config file exists but is not valid JSON. Please check it manually.")
else:
    print(f"✅ Config file doesn't exist yet. Create it at: {config_path}")
    print("   Then add the configuration above.")

print()
print("After updating the config, restart Claude Desktop completely!")

