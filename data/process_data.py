import sys
import pandas as pd
from itertools import chain
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Read message data and categories data into DataFrames and concatenates them into a single DataFrame.
        Returns a list with message_data and categories_data"""
    messages_df = pd.read_csv(messages_filepath, index_col='id')
    categories_df = pd.read_csv(categories_filepath, sep=",|;", skiprows=1, header=None)

    # Getting column names by substracting from the first row all the digits
    # Substract all the digit characters from the categories_df cells
    nested_list_columns = [['id'], list(categories_df.replace(r'-[0-9]', '', regex=True).iloc[0, :][1:])]
    categories_df.columns = list(chain.from_iterable(nested_list_columns))
    categories_df.replace(r'[^0-9]', '', regex=True, inplace=True)
    categories_df = categories_df.astype(int)
    categories_df.set_index('id', inplace=True)

    df = pd.concat([messages_df, categories_df], axis=1)
    return df


def clean_data(df):
    """Drop duplicates from the input (DataFrame)"""
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Saves the df into a sql database called as the argument database_filename"""
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()