
import pandas as pd
# key combination to give format to the code in vs code 
def get_products(path, features): 
    df = pd.read_csv(path)
    df['product_id'] = df['product_id'].astype(str)

    for feature in features:
        df[feature] = df[feature].apply(preprocess_text)

    df['text_data'] = df[features].apply(lambda x: ' '.join(x), axis=1)
    return df


def preprocess_text(text):
    #This function takes a string as input and returns a preprocessed version of the string that is ready for analysis.
    text = str(text).lower() # convert to lowercase
    text = text.replace(',', '') # remove commas
    text = text.replace('.', '') # remove periods
    text = text.replace('-', '') # remove hyphens
    text = text.replace('/', '') # remove slashes
    text = text.replace('&', '') # remove ampersands
    text = text.replace('(', '') # remove left parentheses
    text = text.replace(')', '') # remove right parentheses
    text = text.replace(':', '') # remove colons
    text = text.replace(';', '') # remove semicolons
    return text