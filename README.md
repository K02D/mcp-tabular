# MCP Tabular Data Analysis Server

A Model Context Protocol (MCP) server that provides tools for analyzing numeric and tabular data. Works with CSV files and SQLite databases.

## Features

### Core Tools

| Tool | Description |
|------|-------------|
| `list_data_files` | List available CSV and SQLite files in the data directory |
| `describe_dataset` | Generate statistics for a dataset (shape, types, distributions, missing values) |
| `detect_anomalies` | Find outliers using Z-score or IQR methods |
| `compute_correlation` | Calculate correlation matrices between numeric columns |
| `filter_rows` | Filter data using various operators (eq, gt, lt, contains, etc.) |
| `group_aggregate` | Group data and compute aggregations (sum, mean, count, etc.) |
| `query_sqlite` | Execute SQL queries on SQLite databases |
| `list_tables` | List all tables and schemas in a SQLite database |

### Analytics Tools

| Tool | Description |
|------|-------------|
| `create_pivot_table` | Create Excel-style pivot tables with flexible aggregations |
| `data_quality_report` | Data quality assessment with scores and recommendations |
| `analyze_time_series` | Time series analysis with trends, seasonality, and moving averages |
| `generate_chart` | Create visualizations (bar, line, scatter, histogram, pie, box plots) |
| `merge_datasets` | Join/merge two datasets together (inner, left, right, outer joins) |
| `statistical_test` | Hypothesis testing (t-test, ANOVA, chi-squared, correlation tests) |
| `auto_insights` | Discover patterns and insights |
| `export_data` | Export filtered/transformed data to new CSV files |

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

1. Locate your Claude Desktop config file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Add this configuration (replace `/Users/kirondeb/mcp-tabular` with your actual path):

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

3. Restart Claude Desktop (quit and reopen)

4. Test by asking Claude: "Describe the dataset in data/sample_sales.csv"

See [CONNECT_TO_CLAUDE_DESKTOP.md](CONNECT_TO_CLAUDE_DESKTOP.md) for detailed instructions and troubleshooting.

See [TEST_PROMPTS.md](TEST_PROMPTS.md) for example prompts.

### Sample Data

The project includes sample data for testing:

- `data/sample_sales.csv` - Sales transaction data
- `data/sample.db` - SQLite database with customers, orders, and products tables

To create the SQLite sample database:

```bash
python scripts/create_sample_db.py
```

## Path Resolution

All file paths are resolved relative to the project root directory:
- Relative paths like `data/sample_sales.csv` work from any working directory
- Absolute paths also work as expected
- Paths resolve relative to where `mcp_tabular` is installed

## Tool Examples

### List Data Files

List available data files:

```
list_data_files()
```

Lists all CSV and SQLite files in the data directory with metadata.

### Describe Dataset

Generate statistics for a dataset:

```
describe_dataset(file_path="data/sample_sales.csv")
```

Includes shape, column types, numeric statistics (mean, std, median, skew, kurtosis), categorical value counts, and a sample preview.

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

Calculate correlations between numeric columns:

```
compute_correlation(
    file_path="data/sample_sales.csv",
    method="pearson"
)
```

Includes full correlation matrix and top correlations ranked by strength.

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

List tables and schemas in a SQLite database:

```
list_tables(db_path="data/sample.db")
```

## Advanced Analytics Examples

### Create Pivot Table

Create Excel-style pivot tables:

```
create_pivot_table(
    file_path="data/sample_sales.csv",
    index=["region"],
    columns=["category"],
    values="total_sales",
    aggfunc="sum"
)
```

### Data Quality Report

Generate a data quality assessment:

```
data_quality_report(file_path="data/sample_sales.csv")
```

Includes completeness score, duplicate detection, outlier analysis, and an overall quality grade (A-F).

### Time Series Analysis

Analyze trends and seasonality:

```
analyze_time_series(
    file_path="data/sample_sales.csv",
    date_column="order_date",
    value_column="total_sales",
    freq="M",
    include_forecast=True
)
```

### Generate Charts

Create visualizations (returned as base64 images):

```
generate_chart(
    file_path="data/sample_sales.csv",
    chart_type="bar",
    x_column="category",
    y_column="total_sales",
    title="Sales by Category"
)
```

Supported chart types: `bar`, `line`, `scatter`, `histogram`, `pie`, `box`

### Merge Datasets

Join or merge two datasets:

```
merge_datasets(
    file_path_left="data/orders.csv",
    file_path_right="data/customers.csv",
    on=["customer_id"],
    how="left"
)
```

### Statistical Testing

Run hypothesis tests:

```
statistical_test(
    file_path="data/sample_sales.csv",
    test_type="ttest_ind",
    column1="total_sales",
    group_column="region",
    alpha=0.05
)
```

Supported tests: `ttest_ind`, `ttest_paired`, `chi_squared`, `anova`, `mann_whitney`, `pearson`, `spearman`

### Auto Insights

Discover patterns and insights:

```
auto_insights(file_path="data/sample_sales.csv")
```

Includes insights about correlations, outliers, skewed distributions, missing data, and more.

### Export Data

Export filtered data to a new CSV:

```
export_data(
    file_path="data/sample_sales.csv",
    output_name="electronics_sales",
    filter_column="category",
    filter_operator="eq",
    filter_value="Electronics",
    sort_by="total_sales",
    sort_ascending=False
)
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

