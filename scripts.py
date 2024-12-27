import django
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'recommend.settings'
django.setup()
import pandas as pd

# import csv
# csv_file_path = 'flipKart_com-ecommerce_sample.csv'
# from home.models import *

# with open(csv_file_path, mode='r' , encoding='utf-8') as file:
#     reader = csv.DictReader(file)

#     for row in reader:
#         try:
#             product_name = row['product_name']
#             product_image = eval(row['image'])[0]
#             description = row['description']
#             category = row['product_category_tree'].split('>>')[0].strip('[]"')
#             price = row['retail_price']

#             print(
#                 product_name,
#                 product_image, 
#                 description, 
#                 category, 
#                 price, 
#             )


#             Product.objects.update_or_create(
#                 name=product_name,
#                 defaults={
#                     'product_image' : product_image,
#                     'description' : description,
#                     'category' : category,
#                     'price' : price
#                 }
#             )
#         except Exception as e:
#             print(e)

# print("Products imported successfully!")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from home.models import Product

def get_similar_products(product_id , top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english")
    product_descriptions = Product.objects.all().values_list('description', flat=True)
    tfid_matrix = vectorizer.fit_transform(product_descriptions)
    target_product = Product.objects.get(id = product_id)
    all_products = list(Product.objects.all())
    target_index = all_products.index(target_product)
    cosine_sim = cosine_similarity(tfid_matrix[target_index], tfid_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    similar_indices = [ i for i in similar_indices if i != target_index]
    similar_products = []
    for idx in similar_indices:
        similar_products.append(all_products[idx])

    return similar_products



print(get_similar_products(2616))