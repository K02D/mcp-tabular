# Test Prompts for MCP Tabular

Try these prompts in Claude Desktop to explore the MCP server's capabilities:

## Basic Analysis

- *"List all available data files"*
- *"Describe the dataset in data/sample_sales.csv"*
- *"What's the data quality of my sales dataset?"*
- *"Give me auto insights on data/sample_sales.csv"*
- *"Find anomalies in the total_sales column"*

## Business Analytics

- *"Create a pivot table showing sales by region and category"*
- *"Show me sales trends over time from the sales data"*
- *"Generate a bar chart of total sales by category"*
- *"What are the top correlations in my sales data?"*
- *"Compare sales between Electronics and Home categories using a statistical test"*

## Data Exploration

- *"Filter the sales data to show only Electronics category"*
- *"Group sales by region and show the sum and average"*
- *"Show me a histogram of the total_sales distribution"*
- *"What day of the week has the highest sales?"*
- *"Export all Electronics sales sorted by total_sales descending to a new file"*

## Advanced Analysis

- *"Perform a time series analysis on sales data with date column 'date' and value column 'total_sales'"*
- *"Run a chi-squared test between category and region columns"*
- *"Create a scatter plot of quantity vs total_sales"*
- *"What's the statistical significance of the difference in sales between regions?"*
- *"Show me a pie chart of sales by region"*

## Database Queries

- *"List all tables in data/sample.db"*
- *"Query the sample database to show customers with lifetime value over 1000"*
- *"What's the schema of the orders table?"*

## Tips

- All prompts work with natural language - Claude will automatically call the appropriate MCP tools
- File paths can be relative (like `data/sample_sales.csv`) or absolute
- For charts, Claude will display the base64-encoded images directly in the conversation
- Export operations save files to the `data/` directory with timestamps

