
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import numpy
import glob
from datetime import datetime

# Code for ETL operations on Country-GDP data

# Importing the required libraries

def extract(url, table_attribs):
    ''' This function extracts the required
    information from the website and saves it to a dataframe. The
    function returns the dataframe for further processing. '''

    df = pd.DataFrame(columns=table_attribs)
    
    html_page = requests.get(url).text
    data = BeautifulSoup(html_page, "html.parser")
    
    
    heading = data.find('span', {'id': 'By_market_capitalization'})
    if heading:
        table = heading.find_next('table')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                col = row.find_all('td')
                if len(col)!=0:
                    data_dict = {
                         table_attribs[0]: col[1].find_all('a')[1].contents[0],
                         table_attribs[1]: col[2].contents[0].replace('\n', '')
                         }
                    df1 = pd.DataFrame(data_dict, index=[0])
                    df = pd.concat([df,df1], ignore_index=True)

    return df


def transform(data_frame, csv_path):
    ''' This function converts the Market Capitalization information from USD (Billions) 
    to GBP, EUR, and INR (Billions) based on the exchange rate information in the CSV file. 
    The function returns the transformed dataframe with rounded values to 2 decimal places.'''
    
    exchange_rates = pd.read_csv(csv_path)
    
    # Convert the Market Capitalization to GBP, EUR, and INR
    data_frame["MC_GBP_Billion"] = data_frame['MC_USD_Billion'].astype(float) * exchange_rates[exchange_rates['Currency'] == 'GBP']['Rate'].values[0]
    data_frame["MC_EUR_Billion"] = data_frame["MC_USD_Billion"].astype(float) * exchange_rates[exchange_rates['Currency'] == 'EUR']['Rate'].values[0]
    data_frame["MC_INR_Billion"] = data_frame["MC_USD_Billion"].astype(float) * exchange_rates[exchange_rates['Currency'] == 'INR']['Rate'].values[0]

    # Round the values to 2 decimal places
    data_frame["MC_GBP_Billion"] = data_frame["MC_GBP_Billion"].round(2)
    data_frame["MC_EUR_Billion"] = data_frame["MC_EUR_Billion"].round(2)
    data_frame["MC_INR_Billion"] = data_frame["MC_INR_Billion"].round(2)

    return data_frame

def load_to_csv(df, csv_path):
    ''' This function saves the final dataframe as a `CSV` file 
    in the provided path. Function returns nothing.'''
    
    df.to_csv(csv_path)
    
    

def load_to_db(df, sql_connection, table_name):
    ''' This function saves the final dataframe as a database table
    with the provided name. Function returns nothing.'''
    df.to_sql(table_name, sql_connection, if_exists = 'replace', index = False)

def run_query(query_statement, sql_connection):
    ''' This function runs the stated query on the database table and
    prints the output on the terminal. Function returns nothing. '''
    df = pd.read_sql(query_statement, sql_connection)
    print(f"run_query {query_statement} and result\n {df}")

def log_progress(message):
    ''' This function logs the mentioned message at a given stage of the code execution to a log file. Function returns nothing'''
    timestamp_format = '%Y-%h-%d-%H:%M:%S'
    now = datetime.now()
    timestamp = now.strftime(timestamp_format)
    with open('code_log.txt', "a") as f:
        f.write(timestamp + ', ' + message + '\n') 


if __name__ == '__main__':

    url = 'https://web.archive.org/web/20230908091635 /https://en.wikipedia.org/wiki/List_of_largest_banks'
    csv_src_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-PY0221EN-Coursera/labs/v2/exchange_rate.csv"
    table_attribs = ["Name", "MC_USD_Billion"]
    db_name = 'Banks.db'
    

    log_progress("ETL Job Started")

    log_progress("Extract phase Started")
    extracted_data = extract(url, table_attribs)
    print(f"extracted_data\n{extracted_data}")
    log_progress("Extracted phase Ended")

    log_progress("Transform Job Started")
    # table_attribs_new = ["Name", "MC_USD_Billion", "MC_GBP_Billion", "MC_EUR_Billion", "MC_INR_Billion"]
    trasnsformed_data = transform(extracted_data, csv_src_path)
    print(f"trasnsformed_data\n{trasnsformed_data}")
    log_progress("Transform Job Ended")

    csv_dst_name = "Largest_banks_data.csv"
    
    log_progress("Load Job Started")
    load_to_csv(trasnsformed_data, csv_dst_name)
    sql_connection = sqlite3.connect(db_name)
    table_name = 'Largest_banks'
    load_to_db(trasnsformed_data, sql_connection, table_name)
    log_progress("Load Job Ended")
    
    run_query(f"SELECT * FROM {table_name}", sql_connection)
    
    run_query(f"SELECT AVG(MC_GBP_Billion) FROM {table_name}", sql_connection)
    
    run_query(f"SELECT Name from {table_name} LIMIT 5", sql_connection)
    
    log_progress("ETL Job Eneded")