"""
MCP Server for Tabular Data Analysis.

Provides tools for:
- Dataset description and statistics
- Anomaly detection
- Correlation computation
- Row filtering
- SQLite querying
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from scipy import stats

# Initialize MCP server
mcp = FastMCP(
    "Tabular Data Analysis",
    dependencies=["pandas", "numpy", "scipy"],
)

# Store loaded datasets in memory for efficient re-use
_datasets: dict[str, pd.DataFrame] = {}

# Get project root directory (parent of src/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_path(file_path: str) -> Path:
    """
    Resolve file path relative to project root if it's a relative path.
    
    Args:
        file_path: Absolute or relative file path
    
    Returns:
        Resolved absolute Path
    """
    path = Path(file_path)
    
    # If absolute path, use as-is
    if path.is_absolute():
        return path
    
    # Otherwise, resolve relative to project root
    resolved = _PROJECT_ROOT / path
    return resolved.resolve()


def _load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV or SQLite file."""
    path = _resolve_path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Resolved to: {path}\n"
            f"Project root: {_PROJECT_ROOT}\n"
            f"Current working directory: {Path.cwd()}"
        )
    
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        return pd.read_csv(str(path))
    elif suffix in (".db", ".sqlite", ".sqlite3"):
        # For SQLite, list tables or load first table
        conn = sqlite3.connect(str(path))
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )
        if tables.empty:
            conn.close()
            raise ValueError(f"No tables found in SQLite database: {file_path}")
        first_table = tables.iloc[0]["name"]
        df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
        conn.close()
        return df
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .db/.sqlite")


def _get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Get list of numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


@mcp.tool()
def describe_dataset(file_path: str, include_all: bool = False) -> dict[str, Any]:
    """
    Generate comprehensive statistics for a tabular dataset.
    
    Args:
        file_path: Path to CSV or SQLite file
        include_all: If True, include statistics for all columns (not just numeric)
    
    Returns:
        Dictionary containing:
        - shape: (rows, columns)
        - columns: List of column names with their types
        - numeric_stats: Descriptive statistics for numeric columns
        - missing_values: Count of missing values per column
        - sample: First 5 rows as preview
    """
    df = _load_data(file_path)
    
    # Basic info
    result = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {
            col: str(df[col].dtype) for col in df.columns
        },
        "missing_values": df.isnull().sum().to_dict(),
    }
    
    # Numeric statistics
    numeric_cols = _get_numeric_columns(df)
    if numeric_cols:
        stats_df = df[numeric_cols].describe()
        # Add additional stats
        stats_df.loc["median"] = df[numeric_cols].median()
        stats_df.loc["skew"] = df[numeric_cols].skew()
        stats_df.loc["kurtosis"] = df[numeric_cols].kurtosis()
        result["numeric_stats"] = stats_df.to_dict()
    
    # Categorical columns info
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        result["categorical_columns"] = {
            col: {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
            for col in cat_cols
        }
    
    # Sample data
    result["sample"] = df.head(5).to_dict(orient="records")
    
    return result


@mcp.tool()
def detect_anomalies(
    file_path: str,
    column: str,
    method: str = "zscore",
    threshold: float = 3.0,
) -> dict[str, Any]:
    """
    Detect anomalies/outliers in a numeric column.
    
    Args:
        file_path: Path to CSV or SQLite file
        column: Name of the numeric column to analyze
        method: Detection method - 'zscore' (default), 'iqr', or 'isolation_forest'
        threshold: Threshold for anomaly detection (default 3.0 for zscore, 1.5 for IQR)
    
    Returns:
        Dictionary containing:
        - method: Detection method used
        - anomaly_count: Number of anomalies found
        - anomaly_indices: Row indices of anomalies
        - anomalies: The anomalous rows
        - statistics: Column statistics
    """
    df = _load_data(file_path)
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' is not numeric")
    
    col_data = df[column].dropna()
    
    if method == "zscore":
        # Z-score method
        z_scores = np.abs(stats.zscore(col_data))
        anomaly_mask = z_scores > threshold
        anomaly_indices = col_data[anomaly_mask].index.tolist()
        
    elif method == "iqr":
        # Interquartile Range method
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        anomaly_mask = (col_data < lower_bound) | (col_data > upper_bound)
        anomaly_indices = col_data[anomaly_mask].index.tolist()
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'")
    
    anomalies_df = df.loc[anomaly_indices]
    
    return {
        "method": method,
        "threshold": threshold,
        "column": column,
        "anomaly_count": len(anomaly_indices),
        "anomaly_percentage": round(len(anomaly_indices) / len(col_data) * 100, 2),
        "anomaly_indices": anomaly_indices,
        "anomalies": anomalies_df.to_dict(orient="records"),
        "statistics": {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "median": float(col_data.median()),
        }
    }


@mcp.tool()
def compute_correlation(
    file_path: str,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> dict[str, Any]:
    """
    Compute correlation matrix between numeric columns.
    
    Args:
        file_path: Path to CSV or SQLite file
        columns: List of columns to include (default: all numeric columns)
        method: Correlation method - 'pearson' (default), 'spearman', or 'kendall'
    
    Returns:
        Dictionary containing:
        - method: Correlation method used
        - correlation_matrix: Full correlation matrix
        - top_correlations: Top 10 strongest correlations (excluding self-correlations)
    """
    df = _load_data(file_path)
    
    # Get numeric columns
    if columns:
        # Validate provided columns
        invalid = [c for c in columns if c not in df.columns]
        if invalid:
            raise ValueError(f"Columns not found: {invalid}")
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation")
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Find top correlations (excluding diagonal)
    correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Upper triangle only
                corr_value = corr_matrix.loc[col1, col2]
                if not np.isnan(corr_value):
                    correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(float(corr_value), 4),
                        "strength": _interpret_correlation(abs(corr_value))
                    })
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    return {
        "method": method,
        "columns_analyzed": corr_matrix.columns.tolist(),
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "top_correlations": correlations[:10],
    }


def _interpret_correlation(value: float) -> str:
    """Interpret correlation strength."""
    if value >= 0.9:
        return "very_strong"
    elif value >= 0.7:
        return "strong"
    elif value >= 0.5:
        return "moderate"
    elif value >= 0.3:
        return "weak"
    else:
        return "negligible"


@mcp.tool()
def filter_rows(
    file_path: str,
    column: str,
    operator: str,
    value: str | float | int,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Filter rows based on a condition.
    
    Args:
        file_path: Path to CSV or SQLite file
        column: Column name to filter on
        operator: Comparison operator - 'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'contains', 'startswith', 'endswith'
        value: Value to compare against
        limit: Maximum number of rows to return (default 100)
    
    Returns:
        Dictionary containing:
        - filter_applied: Description of the filter
        - original_count: Number of rows before filtering
        - filtered_count: Number of rows after filtering
        - rows: Filtered rows (up to limit)
    """
    df = _load_data(file_path)
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
    
    original_count = len(df)
    
    # Apply filter based on operator
    if operator == "eq":
        mask = df[column] == value
    elif operator == "ne":
        mask = df[column] != value
    elif operator == "gt":
        mask = df[column] > float(value)
    elif operator == "gte":
        mask = df[column] >= float(value)
    elif operator == "lt":
        mask = df[column] < float(value)
    elif operator == "lte":
        mask = df[column] <= float(value)
    elif operator == "contains":
        mask = df[column].astype(str).str.contains(str(value), case=False, na=False)
    elif operator == "startswith":
        mask = df[column].astype(str).str.startswith(str(value), na=False)
    elif operator == "endswith":
        mask = df[column].astype(str).str.endswith(str(value), na=False)
    else:
        raise ValueError(
            f"Unknown operator: {operator}. Use: eq, ne, gt, gte, lt, lte, contains, startswith, endswith"
        )
    
    filtered_df = df[mask]
    
    return {
        "filter_applied": f"{column} {operator} {value}",
        "original_count": original_count,
        "filtered_count": len(filtered_df),
        "rows": filtered_df.head(limit).to_dict(orient="records"),
        "truncated": len(filtered_df) > limit,
    }


@mcp.tool()
def query_sqlite(
    db_path: str,
    query: str,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Execute a SQL query on a SQLite database.
    
    Args:
        db_path: Path to SQLite database file
        query: SQL query to execute (SELECT queries only for safety)
        limit: Maximum number of rows to return (default 100)
    
    Returns:
        Dictionary containing:
        - query: The executed query
        - row_count: Number of rows returned
        - columns: List of column names
        - rows: Query results
    """
    # Basic safety check - only allow SELECT
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed for safety")
    
    path = _resolve_path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            f"Resolved to: {path}\n"
            f"Project root: {_PROJECT_ROOT}"
        )
    
    conn = sqlite3.connect(str(path))
    try:
        # Add LIMIT if not present
        if "LIMIT" not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn)
        
        return {
            "query": query,
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "rows": df.to_dict(orient="records"),
        }
    finally:
        conn.close()


@mcp.tool()
def list_tables(db_path: str) -> dict[str, Any]:
    """
    List all tables in a SQLite database.
    
    Args:
        db_path: Path to SQLite database file
    
    Returns:
        Dictionary containing table names and their schemas
    """
    path = _resolve_path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            f"Resolved to: {path}\n"
            f"Project root: {_PROJECT_ROOT}"
        )
    
    conn = sqlite3.connect(str(path))
    try:
        # Get table names
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )
        
        result = {"tables": {}}
        
        for table_name in tables["name"]:
            # Get schema for each table
            schema = pd.read_sql_query(
                f"PRAGMA table_info({table_name})", conn
            )
            
            # Get row count
            count = pd.read_sql_query(
                f"SELECT COUNT(*) as cnt FROM {table_name}", conn
            ).iloc[0]["cnt"]
            
            result["tables"][table_name] = {
                "row_count": int(count),
                "columns": [
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "nullable": not row["notnull"],
                        "primary_key": bool(row["pk"]),
                    }
                    for _, row in schema.iterrows()
                ]
            }
        
        return result
    finally:
        conn.close()


@mcp.tool()
def group_aggregate(
    file_path: str,
    group_by: list[str],
    aggregations: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Group data and compute aggregations.
    
    Args:
        file_path: Path to CSV or SQLite file
        group_by: Columns to group by
        aggregations: Dictionary mapping column names to list of aggregation functions
                     (e.g., {"sales": ["sum", "mean"], "quantity": ["count", "max"]})
                     Supported: sum, mean, median, min, max, count, std, var
    
    Returns:
        Dictionary containing grouped and aggregated data
    """
    df = _load_data(file_path)
    
    # Validate group_by columns
    invalid = [c for c in group_by if c not in df.columns]
    if invalid:
        raise ValueError(f"Group-by columns not found: {invalid}")
    
    # Validate aggregation columns
    for col in aggregations:
        if col not in df.columns:
            raise ValueError(f"Aggregation column '{col}' not found")
    
    # Perform groupby
    grouped = df.groupby(group_by).agg(aggregations)
    
    # Flatten column names
    grouped.columns = ["_".join(col).strip() for col in grouped.columns]
    grouped = grouped.reset_index()
    
    return {
        "group_by": group_by,
        "aggregations": aggregations,
        "group_count": len(grouped),
        "result": grouped.to_dict(orient="records"),
    }


@mcp.tool()
def list_data_files(data_dir: str = "data") -> dict[str, Any]:
    """
    List available data files in the project data directory.
    
    Args:
        data_dir: Relative path to data directory (default: "data")
    
    Returns:
        Dictionary containing list of available CSV and SQLite files
    """
    data_path = _resolve_path(data_dir)
    
    if not data_path.exists():
        return {
            "data_directory": str(data_path),
            "exists": False,
            "files": []
        }
    
    csv_files = []
    db_files = []
    
    for file_path in sorted(data_path.iterdir()):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            file_info = {
                "name": file_path.name,
                "path": str(file_path.relative_to(_PROJECT_ROOT)),
                "size_bytes": file_path.stat().st_size,
            }
            
            if suffix == ".csv":
                # Try to get basic info about CSV
                try:
                    df = pd.read_csv(str(file_path), nrows=0)
                    file_info["columns"] = df.columns.tolist()
                    file_info["column_count"] = len(df.columns)
                except Exception:
                    pass
                csv_files.append(file_info)
            elif suffix in (".db", ".sqlite", ".sqlite3"):
                db_files.append(file_info)
    
    return {
        "data_directory": str(data_path.relative_to(_PROJECT_ROOT)),
        "absolute_path": str(data_path),
        "csv_files": csv_files,
        "sqlite_files": db_files,
        "total_files": len(csv_files) + len(db_files),
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

