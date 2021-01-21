import pandas as pd


def get_sales_data():
    """Superstore data
    From https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls
    Used in Blog Post: https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
    This blog post filters for 'furniture'
    """
    return pd.read_excel("time_series/resources/sample _superstore.xls")
