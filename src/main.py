from secData import get_data

if __name__ == '__main__':
    txt = get_data(
        'AAPL',
        year="2022",
        filing_type="10-Q",
        quarters="ALL",
        include_amends=True,
        num_workers=1
    )
    print(txt)