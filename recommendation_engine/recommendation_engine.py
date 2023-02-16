
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import data_processing

# Get products dataset
features = ['category', 'title', 'description_short', 'description_general']
path='data/products.csv'
products_df = data_processing.get_products(path, features)

def get_top_similar(product_id, products_df, tfidf_matrix, n=5, category=None):
    # get the index position of the input product
    product_index = products_df.index[products_df['product_id'] == product_id][0]
    
    # compute the similarity scores between the input product and all other products
    similarity_scores = cosine_similarity(tfidf_matrix[product_index], tfidf_matrix)
    
    # get the top n similar products
    similar_indices = similarity_scores.argsort()[0][-n-1:-1][::-1]
    
    # filter out products in different categories (if category is specified)
    if category is not None:
        category_indices = products_df[products_df['category'] == category].index.values
        similar_indices = [idx for idx in similar_indices if idx in category_indices]
    
    # get the product IDs and similarity scores of the top similar products
    similar_products = [{'product_id': products_df.loc[idx, 'product_id'], 'similarity': similarity_scores[0][idx]} for idx in similar_indices]
    
    return similar_products

def create_tfidf_matrix(text_data):
    # create a TF-IDF vectorizer and fit it to the text data
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text_data)
    
    # transform the text data into a TF-IDF matrix
    tfidf_matrix = vectorizer.transform(text_data)
    
    return tfidf_matrix


# create a TF-IDF matrix based on the text data in the products dataframe
tfidf_matrix = create_tfidf_matrix(products_df['text_data'])

# compute the top similar products to a given product ID
product_id = '13599'
similar_products = get_top_similar(product_id, products_df, tfidf_matrix, n=1)

print(similar_products)
