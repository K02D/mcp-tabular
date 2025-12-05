# Connecting MCP Tabular Server to Claude Desktop

## Step-by-Step Instructions

### 1. Locate Claude Desktop Configuration File

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### 2. Open or Create the Configuration File

If the file doesn't exist, create it. If it exists, you'll need to merge the configuration.

### 3. Add the MCP Server Configuration

Add this configuration to your `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "tabular-data": {
      "command": "/Users/kirondeb/mcp-tabular/.venv/bin/python",
      "args": [
        "-m",
        "mcp_tabular.server"
      ]
    }
  }
}
```

**Important:** Replace `/Users/kirondeb/mcp-tabular` with the actual path to your `mcp-tabular` directory.

### 4. If You Already Have Other MCP Servers

If you already have other MCP servers configured, merge them like this:

```json
{
  "mcpServers": {
    "existing-server": {
      "command": "...",
      "args": [...]
    },
    "tabular-data": {
      "command": "/Users/kirondeb/mcp-tabular/.venv/bin/python",
      "args": [
        "-m",
        "mcp_tabular.server"
      ]
    }
  }
}
```

### 5. Restart Claude Desktop

After saving the configuration file:
1. **Quit Claude Desktop completely** (not just close the window)
2. **Reopen Claude Desktop**
3. The MCP server should now be connected

### 6. Verify Connection

Once Claude Desktop restarts, you should see:
- The MCP server listed in Claude Desktop's settings
- Tools available when chatting with Claude

You can test it by asking Claude:
- "Describe the dataset in data/sample_sales.csv"
- "Find anomalies in the total_sales column"
- "What tables are in the sample.db database?"

## Alternative Configuration Options

### Option 1: Using System Python (if installed globally)

If you installed the package globally with `pip install -e .`:

```json
{
  "mcpServers": {
    "tabular-data": {
      "command": "mcp-tabular"
    }
  }
}
```

### Option 2: Using uv (if you have uv installed)

```json
{
  "mcpServers": {
    "tabular-data": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/kirondeb/mcp-tabular",
        "run",
        "mcp-tabular"
      ]
    }
  }
}
```

### Option 3: Using Full Python Path

```json
{
  "mcpServers": {
    "tabular-data": {
      "command": "/usr/local/bin/python3",
      "args": [
        "-m",
        "mcp_tabular.server"
      ],
      "env": {
        "PYTHONPATH": "/Users/kirondeb/mcp-tabular/src"
      }
    }
  }
}
```

## Troubleshooting

### Server Not Connecting

1. **Check the Python path is correct:**
   ```bash
   /Users/kirondeb/mcp-tabular/.venv/bin/python -m mcp_tabular.server
   ```
   This should start the server without errors.

2. **Check file permissions:**
   Make sure the configuration file is readable and the Python executable has execute permissions.

3. **Check Claude Desktop logs:**
   Look for error messages in Claude Desktop's console/logs.

### Tools Not Appearing

1. Make sure you restarted Claude Desktop completely
2. Check that the server is running (you should see it in Claude Desktop's MCP settings)
3. Try asking Claude directly: "What MCP tools do you have available?"

### Path Issues

If you get "command not found" errors:
- Use absolute paths (starting with `/`)
- Make sure the virtual environment exists at that path
- Verify Python is installed at that location

## Quick Test

After configuration, test the server manually:

```bash
cd /Users/kirondeb/mcp-tabular
source .venv/bin/activate
python -m mcp_tabular.server
```

If this runs without errors, the configuration should work in Claude Desktop.

