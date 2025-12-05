# MCP Tabular Data Analysis Server

A Model Context Protocol (MCP) server that provides powerful tools for analyzing numeric and tabular data. Works with CSV files and SQLite databases.

## Features

### Tools Available

| Tool | Description |
|------|-------------|
| `list_data_files` | List available CSV and SQLite files in the data directory |
| `describe_dataset` | Generate comprehensive statistics for a dataset (shape, types, distributions, missing values) |
| `detect_anomalies` | Find outliers using Z-score or IQR methods |
| `compute_correlation` | Calculate correlation matrices between numeric columns |
| `filter_rows` | Filter data using various operators (eq, gt, lt, contains, etc.) |
| `group_aggregate` | Group data and compute aggregations (sum, mean, count, etc.) |
| `query_sqlite` | Execute SQL queries on SQLite databases |
| `list_tables` | List all tables and schemas in a SQLite database |

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install with uv

```bash
cd mcp-tabular
uv sync
```

### Install with pip

```bash
cd mcp-tabular
pip install -e .
```

## Usage

### Running the Server Directly

```bash
# With uv
uv run mcp-tabular

# With pip installation
mcp-tabular
```

### Configure with Claude Desktop

**Quick Setup:**

1. **Find your Claude Desktop config file:**
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. **Add this configuration** (replace `/Users/kirondeb/mcp-tabular` with your actual path):

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

3. **Restart Claude Desktop completely** (quit and reopen)

4. **Test it** by asking Claude: "Describe the dataset in data/sample_sales.csv"

📖 **For detailed instructions and troubleshooting**, see [CONNECT_TO_CLAUDE_DESKTOP.md](CONNECT_TO_CLAUDE_DESKTOP.md)

### Sample Data

The project includes sample data for testing:

- `data/sample_sales.csv` - Sales transaction data
- `data/sample.db` - SQLite database with customers, orders, and products tables

To create the SQLite sample database:

```bash
python scripts/create_sample_db.py
```

## Path Resolution

**Important:** All file paths are resolved relative to the project root directory. This means:
- Relative paths like `data/sample_sales.csv` work from any working directory
- Absolute paths also work as expected
- The server automatically resolves paths relative to where `mcp_tabular` is installed

## Tool Examples

### List Data Files

Discover available data files:

```
list_data_files()
```

Returns all CSV and SQLite files in the data directory with metadata.

### Describe Dataset

Get comprehensive statistics about a dataset:

```
describe_dataset(file_path="data/sample_sales.csv")
```

Returns shape, column types, numeric statistics (mean, std, median, skew, kurtosis), categorical value counts, and a sample preview.

### Detect Anomalies

Find outliers in numeric columns:

```
detect_anomalies(
    file_path="data/sample_sales.csv",
    column="total_sales",
    method="zscore",
    threshold=3.0
)
```

Supports `zscore` and `iqr` methods.

### Compute Correlation

Analyze relationships between numeric columns:

```
compute_correlation(
    file_path="data/sample_sales.csv",
    method="pearson"
)
```

Returns full correlation matrix and top correlations ranked by strength.

### Filter Rows

Filter data based on conditions:

```
filter_rows(
    file_path="data/sample_sales.csv",
    column="category",
    operator="eq",
    value="Electronics"
)
```

Operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contains`, `startswith`, `endswith`

### Group & Aggregate

Group data and compute aggregations:

```
group_aggregate(
    file_path="data/sample_sales.csv",
    group_by=["category", "region"],
    aggregations={"total_sales": ["sum", "mean"], "quantity": ["count"]}
)
```

### Query SQLite

Execute SQL queries on databases:

```
query_sqlite(
    db_path="data/sample.db",
    query="SELECT * FROM customers WHERE lifetime_value > 1000"
)
```

### List Tables

Explore SQLite database structure:

```
list_tables(db_path="data/sample.db")
```

## Development

### Run Tests

```bash
uv run pytest
```

### Project Structure

```
mcp-tabular/
├── src/
│   └── mcp_tabular/
│       ├── __init__.py
│       └── server.py      # Main MCP server implementation
├── data/
│   ├── sample_sales.csv   # Sample CSV data
│   └── sample.db          # Sample SQLite database
├── scripts/
│   └── create_sample_db.py
├── pyproject.toml
├── claude_desktop_config.json
└── README.md
```

## License

MIT

