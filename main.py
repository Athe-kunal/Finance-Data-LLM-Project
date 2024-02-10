from src.vectorDatabase import create_database
from src.colbert import build_index, query_data

if __name__ == "__main__":
    build_index(ticker="AAPL", year=2023)
