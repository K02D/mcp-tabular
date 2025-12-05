"""Create a sample SQLite database for testing."""

import sqlite3
from pathlib import Path

# Create data directory if needed
data_dir = Path(__file__).parent.parent / "data"
data_dir.mkdir(exist_ok=True)

db_path = data_dir / "sample.db"

# Remove existing database
if db_path.exists():
    db_path.unlink()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create customers table
cursor.execute("""
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    city TEXT,
    signup_date TEXT,
    lifetime_value REAL
)
""")

# Create orders table
cursor.execute("""
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date TEXT,
    total_amount REAL,
    status TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
)
""")

# Create products table
cursor.execute("""
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price REAL,
    stock_quantity INTEGER
)
""")

# Insert sample customers
customers = [
    (1, "Alice Johnson", "alice@example.com", "New York", "2023-01-15", 1250.00),
    (2, "Bob Smith", "bob@example.com", "Los Angeles", "2023-02-20", 890.50),
    (3, "Carol Williams", "carol@example.com", "Chicago", "2023-03-10", 2100.75),
    (4, "David Brown", "david@example.com", "Houston", "2023-04-05", 450.25),
    (5, "Eve Davis", "eve@example.com", "Phoenix", "2023-05-12", 3200.00),
    (6, "Frank Miller", "frank@example.com", "Philadelphia", "2023-06-18", 780.00),
    (7, "Grace Wilson", "grace@example.com", "San Antonio", "2023-07-22", 1560.50),
    (8, "Henry Moore", "henry@example.com", "San Diego", "2023-08-30", 920.25),
    (9, "Ivy Taylor", "ivy@example.com", "Dallas", "2023-09-14", 15000.00),
    (10, "Jack Anderson", "jack@example.com", "San Jose", "2023-10-25", 680.75),
]
cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers)

# Insert sample orders
orders = [
    (1, 1, "2024-01-05", 125.50, "completed"),
    (2, 1, "2024-01-15", 89.99, "completed"),
    (3, 2, "2024-01-08", 250.00, "completed"),
    (4, 3, "2024-01-10", 175.25, "pending"),
    (5, 3, "2024-01-20", 320.00, "completed"),
    (6, 4, "2024-01-12", 45.99, "cancelled"),
    (7, 5, "2024-01-18", 890.00, "completed"),
    (8, 5, "2024-01-25", 450.50, "completed"),
    (9, 6, "2024-01-22", 120.00, "pending"),
    (10, 7, "2024-01-28", 275.75, "completed"),
    (11, 8, "2024-02-01", 180.25, "completed"),
    (12, 9, "2024-02-05", 5500.00, "completed"),
    (13, 9, "2024-02-10", 3200.00, "pending"),
    (14, 10, "2024-02-08", 95.00, "cancelled"),
    (15, 1, "2024-02-12", 210.00, "completed"),
]
cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

# Insert sample products
products = [
    (1, "Laptop Pro", "Electronics", 1299.99, 50),
    (2, "Wireless Mouse", "Electronics", 29.99, 200),
    (3, "USB-C Hub", "Electronics", 49.99, 150),
    (4, "Ergonomic Chair", "Furniture", 399.99, 30),
    (5, "Standing Desk", "Furniture", 599.99, 25),
    (6, "Monitor 27\"", "Electronics", 349.99, 75),
    (7, "Keyboard Mechanical", "Electronics", 129.99, 100),
    (8, "Desk Lamp", "Furniture", 45.99, 80),
    (9, "Webcam HD", "Electronics", 79.99, 120),
    (10, "Headphones Pro", "Electronics", 249.99, 60),
]
cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products)

conn.commit()
conn.close()

print(f"Created sample database at: {db_path}")
print("Tables: customers, orders, products")

