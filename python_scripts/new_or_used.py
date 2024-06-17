"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json
import os
import pandas as pd
import sys
module_path = os.path.abspath(os.path.join('/Users/juanmanuelpaiba/Documents/Juan_Paiba/new_or_used_algorithm_MELI/', 'python_scripts'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utilities_meli # type: ignore
from feature_engine.encoding import RareLabelEncoder

os.chdir(path="/Users/juanmanuelpaiba/Documents/Juan_Paiba/new_or_used_algorithm_MELI")
# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("data/Inputs/MLA_100k_checked_v3.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    # ...

    # Convert X_test and y_test into a DataFrame

df_test = pd.DataFrame(X_test)
df_test['condition'] = y_test
df_productos = df_test.copy()


df_seller = pd.json_normalize(df_productos['seller_address'])
df_seller.rename(columns={'country.name': 'country_name', 'country.id': 'country_id', 'state.name': 
                          'state_name', 'state.id': 'state_id', 'city.name': 'city_name', 'city.id': 'city_id' }, inplace=True)

#Shipping
df_shipping = pd.json_normalize(df_productos['shipping'])
df_shipping = df_shipping.drop(columns=['tag'], errors='ignore')

#attributes
df_attributes = pd.json_normalize(df_productos['attributes'])
df_attr_0 = pd.json_normalize(df_attributes[0])
df_attr_1 = pd.json_normalize(df_attributes[1])
df_attr_0.rename(columns={'value_name': 'season_name'}, inplace=True)
df_attr_0 = df_attr_0.drop(columns=['value_id', 'attribute_group_id', 'name', 'attribute_group_name','id'], errors='ignore')
df_attr_1.rename(columns={'value_name': 'gender_name'}, inplace=True)
df_attr_1 = df_attr_1.drop(columns=['value_id', 'attribute_group_id', 'name', 'attribute_group_name','id'], errors='ignore')

#Mdo Pago
df_mdopgo = pd.json_normalize(df_productos['non_mercado_pago_payment_methods'])
#Ajusto 0 las siguientes hasta 11 elimino por valores > Missing 
df_0 = pd.json_normalize(df_mdopgo[0])
df_0.rename(columns={'description': 'descrip_mdo_0', 'id': 'id_mdo_0', 'type': 'type_mdo_0'}, inplace=True)

#Elimino variables iniciales ajustadas
var_drop = ['seller_address', 'shipping', 'non_mercado_pago_payment_methods', 'pictures','attributes']
df_tabular = df_productos.drop(var_drop, axis=1)

# Concat df_ajust
df_productos_00 = pd.concat([df_tabular, df_seller, df_shipping, df_0, df_attr_0,df_attr_1], axis=1)

# Reemplazar 'new' por 1 y 'used' por 0 en la df_ajust
df_productos_00['target'] = df_productos_00['condition'].replace({'new': 1, 'used': 0})

df_productos_00 = df_productos_00.drop(columns=['sub_status', 'deal_ids', 'variations', 'tags', 'listing_source',
                   'coverage_areas', 'descriptions', 'thumbnail', 'secure_thumbnail', 'methods', 'free_methods'], errors='ignore')

df_productos_00['warranty_cleaned'] = df_productos_00['warranty'].apply(lambda x: utilities_meli.clean_warranty(x))
df_productos_00['warranty_type'] = df_productos_00['warranty'].apply(lambda x: utilities_meli.classify_condition(x))


df_productos_00['first_two_words_title'] = df_productos_00['title'].apply(utilities_meli.extract_first_two_words)
df_productos_00['first_word_title'] = df_productos_00['title'].apply(utilities_meli.extract_first_word)
df_productos_00['first_three_words_title'] = df_productos_00['title'].apply(utilities_meli.extract_first_three_words)
df_productos_00['title_type'] = df_productos_00['title'].apply(lambda x: utilities_meli.classify_condition(x))

rare_values_to_replace = [-2147483648, 11111111, 1111111111, 8888888, 9000000, 123456789, 112111111]

# Replace rare values in 'base_price' column
df_products_00 = utilities_meli.replace_rare_values(df_productos_00, 'base_price', rare_values_to_replace)
df_products_00 = utilities_meli.replace_rare_values(df_products_00, 'base_price', rare_values_to_replace)

# 'date_created' a tipo datetime
df_products_00['date_created_month'] = pd.to_datetime(df_products_00['date_created']).dt.strftime('%Y%m')
df_products_00['date_created'] = pd.to_datetime(df_products_00['date_created'])
# Variables Día - Mes
df_products_00['month'] = df_products_00['date_created'].dt.month
df_products_00['weekday'] = df_products_00['date_created'].dt.weekday
# Variable 'year_month' tipo texto
df_products_00['year_month'] = df_products_00['date_created'].dt.strftime('%Y-%m')
# status
df_products_00['concat_status'] = df_products_00['year_month'] + '_' + df_products_00['status']
# listing_type_id
df_products_00['concat_var_lt'] = df_products_00['year_month'] + '_' + df_products_00['listing_type_id']
# state_id
df_products_00['concat_var_state'] = df_products_00['year_month'] + '_' + df_products_00['state_id']
# automatic_relist
df_products_00['automatic_relist_str'] = df_products_00['automatic_relist'].astype(str)
df_products_00['concat_var_autrelist'] = df_products_00['year_month'] + '_' + df_products_00['automatic_relist_str']
# accepts_mercadopago
df_products_00['accepts_mercadopago_str'] = df_products_00['accepts_mercadopago'].astype(str)
df_products_00['concat_var_accmdopag'] = df_products_00['year_month'] + '_' + df_products_00['accepts_mercadopago_str']
# local_pick_up
df_products_00['local_pick_up_str'] = df_products_00['local_pick_up'].astype(str)
df_products_00['concat_var_localpu'] = df_products_00['year_month'] + '_' + df_products_00['local_pick_up_str']
# free_shipping
df_products_00['free_shipping_str'] = df_products_00['free_shipping'].astype(str)
df_products_00['concat_var_freesh'] = df_products_00['year_month'] + '_' + df_products_00['free_shipping_str']

df_products_00= df_products_00.drop(columns=['permalink','seller_id','warranty','condition','site_id','international_delivery_mode',
                                      'parent_item_id','last_updated','id','title','catalog_product_id',
                                      'dimensions','city_name','stop_time', 'start_time'])

df_products_00 = df_products_00.drop(columns=['differential_pricing','original_price',
                                              'official_store_id','date_created'])


columns_to_group = ['mode', 'status', 'listing_type_id', 'state_id', 'automatic_relist',
                    'accepts_mercadopago', 'local_pick_up', 'free_shipping', 'warranty_cleaned', 'weekday','title_type','warranty_type']

value_column = 'base_price'
value_column_1 = 'initial_quantity'
value_column_2 = 'sold_quantity'

for column in columns_to_group:
    df_products_00 = utilities_meli.calculate_group_stats(df_products_00, column, value_column)
    df_products_00 = utilities_meli.calculate_group_stats(df_products_00, column, value_column_1)
    df_products_00 = utilities_meli.calculate_group_stats(df_products_00, column, value_column_2)


var_dummies =['listing_type_id', 'buying_mode', 'category_id', 'currency_id',
            'status', 'video_id', 'country_name', 'country_id', 'state_name',
            'state_id', 'city_id', 'mode', 'descrip_mdo_0', 'id_mdo_0',
            'type_mdo_0', 'season_name', 'gender_name', 'warranty_cleaned',
            'first_two_words_title', 'first_word_title', 'first_three_words_title',
            'title_type', 'date_created_month', 'year_month', 'concat_status',
            'concat_var_lt', 'concat_var_state', 'automatic_relist_str',
            'concat_var_autrelist', 'accepts_mercadopago_str',
            'concat_var_accmdopag', 'local_pick_up_str', 'concat_var_localpu',
            'free_shipping_str', 'concat_var_freesh']

# groups rare or infrequent categories in a new category called “Rare”, or any other name entered by the user
encoder = RareLabelEncoder(tol=0.03, n_categories=2, variables=var_dummies, replace_with='Rare', missing_values='ignore')

encoder.fit(df_products_00)
# transform the data
df_products_01 = encoder.transform(df_products_00)


###########################################
## Paso Variables Categoricas a Dummies
###########################################
df_products_01 = pd.get_dummies(df_products_01)
# Reemplazar NaN con 0
df_products_01 = df_products_01.fillna(0)
# Convertir True y False a 1 y 0
df_products_01 = df_products_01.astype(int)

df_products_01 = utilities_meli.create_transformed_columns(df_products_01,'price')
df_products_01 = utilities_meli.create_transformed_columns(df_products_01,'initial_quantity')
df_products_01 = utilities_meli.create_transformed_columns(df_products_01,'sold_quantity')
df_products_01 = utilities_meli.create_transformed_columns(df_products_01,'available_quantity')
df_products_01 = utilities_meli.create_transformed_columns(df_products_01,'var_base_price_mode')
df_products_01 = utilities_meli.create_transformed_columns(df_products_01,'var_base_price_title_type')
df_products_01 = utilities_meli.create_transformed_columns(df_products_01,'var_base_price_warranty_cleaned')

df_products_01['price_x_initial_quantity'] = df_products_01['price']*df_products_01['initial_quantity']
df_products_01['price_x_sold_quantity'] = df_products_01['price']*df_products_01['sold_quantity']
df_products_01['price_2_x_sold_quantity'] = df_products_01['price_square']*df_products_01['sold_quantity']
df_products_01['price_x_initial_quantity'] = df_products_01['price']*df_products_01['initial_quantity']
df_products_01['price_x_sold_quantity'] = df_products_01['price']*df_products_01['sold_quantity']
df_products_01['price_4_x_sold_quantity'] = df_products_01['price_fourth']*df_products_01['sold_quantity']
df_products_01['price_4_x_initial_quantity'] = df_products_01['price_fourth']*df_products_01['initial_quantity']

print(df_products_01.shape)

