import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import cv2
import copy
from skimage import io
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle

with open("train_images.pickle", 'rb') as f:
    train_images = pickle.load(f)

with open("test_images.pickle", 'rb') as f:
    test_images = pickle.load(f)
    
train_df = pd.read_csv('data/train.csv', index_col = 'id')
test_df = pd.read_csv('data/test.csv', index_col = 'id')

MIN_GENRE_COUNT = 50
MIN_AUTHORS_COUNT = 4


def number1_extraction(txt):
    try:
        return int(txt[35:45])
    except:
        try:
            return int(txt[70:80])
        except:
            return 0
def number2_extraction(txt):
    try:
        return int(txt[47:-4])
    except:
        try:
            return int(txt[82:90])
        except:
            return 0

### Feature processing (master)
def get_int(txt):
    try:
        return int(re.match(r'(\d+)', txt).group(1))
    except:
        return 0

def url_to_image(url):
    try:
        return cv2.cvtColor(io.imread(url), cv2.COLOR_BGR2RGB)
    except:
        return np.array([[[0]]])

def generate_distribution(df, column = "book_genre"):
    genres = df[column].apply(lambda x: np.array(x.split("|")) if type(x) == str else [])
    
    all_genres = []
    for one_list in genres.values:
        for item in one_list:
            all_genres.append(item)


    genre_distribution = pd.Series(all_genres).value_counts(normalize=False)
    return genre_distribution
    
def generate_distribution(df, column = "book_genre"):
    genres = df[column].apply(lambda x: np.array(x.split("|")) if type(x) == str else [])
    
    all_genres = []
    for one_list in genres.values:
        for item in one_list:
            all_genres.append(item)


    genre_distribution = pd.Series(all_genres).value_counts(normalize=False)
    return genre_distribution

genres_train_dist =  generate_distribution(train_df,column = "book_genre")
authors_train_dist =  generate_distribution(train_df,column = "book_authors")   
genres_test_dist =  generate_distribution(test_df,column = "book_genre")
authors_test_dist =  generate_distribution(test_df,column = "book_authors")
 
genr_train =set(list(genres_train_dist[genres_train_dist >= MIN_GENRE_COUNT].index))
genr_test = set(list(genres_test_dist[genres_test_dist >=MIN_GENRE_COUNT].index))

authors_test = set(list(authors_test_dist[authors_test_dist >=MIN_AUTHORS_COUNT].index))
authors_train =set(list(authors_train_dist[authors_train_dist >= MIN_AUTHORS_COUNT].index))

good_genres = genr_train.intersection(genr_test)
good_authors = authors_train.intersection(authors_test)


def one_hot_genres(df, good_genres,column = "book_genre"):
    genres = df[column].apply(lambda x: np.array(x.split("|")) if type(x) == str else [])
    genres_dict = {genre: 0 for genre in good_genres}
    
    
    res_genres = []
    for row_genres in genres:
        row_genre_dict = copy.deepcopy(genres_dict)
        for g in row_genres:
            if g in good_genres:
                row_genre_dict[g] = 1
            else:
                row_genre_dict['Other_'+column] = 1
        res_genres.append(row_genre_dict)
    
    genre_df = pd.DataFrame(res_genres)
    
    for g in genre_df.columns:
        df[g] = genre_df[g].fillna(0)

    return df


def preprocessing(df):
    df['book_pages'] = df['book_pages'].apply(get_int) 
    df['book_desc_len'] = df['book_desc'].apply(len) 
    df['ratio_review_rating'] =  df['book_review_count'] / df['book_rating_count']
    df['ratio_rating_review'] =  df['book_rating_count'] / df['book_review_count']
    df['n1'] = df['book_image_url'].apply(number1_extraction)
    df['n2'] = df['book_image_url'].apply(number2_extraction)
    
    df = one_hot_genres(df, good_genres, column = "book_genre")
    df = one_hot_genres(df, good_authors, column = "book_authors")
    
    return df

def img_features(df, images):
    df['image_width'] = images.apply(lambda img: img.shape[0])
    df['image_heigth'] = images.apply(lambda img: img.shape[1])
    df['image_size_product'] = df['image_width'] * df['image_heigth']
    
    return df

train_df = preprocessing(train_df)
test_df = preprocessing(test_df)

train_df = img_features(train_df, train_images)
test_df = img_features(test_df, test_images)

not_norm_features = ['book_title', 'book_image_url', 'book_desc', 'book_genre',
       'book_authors', 'book_format', 'book_rating']


from sklearn.preprocessing import MinMaxScaler
import pickle

train_df = train_df.replace([np.inf, -np.inf], 1)
test_df = test_df.replace([np.inf, -np.inf], 1)

scaler = MinMaxScaler()

num_df_train = train_df[[col for col in train_df.columns if col not in not_norm_features]]
num_df_test = test_df[[col for col in test_df.columns if col not in not_norm_features]]

num_df_train = num_df_train.fillna(0)
num_df_test = num_df_test.fillna(0)

scaler = scaler.fit(num_df_train)

columns = num_df_train.columns
norm_num_df_train = scaler.transform(num_df_train)
norm_num_df_test = scaler.transform(num_df_test)

norm_num_df_train = pd.DataFrame(norm_num_df_train, columns=columns)
norm_num_df_test = pd.DataFrame(norm_num_df_test, columns=columns)

scaler = {'scaler': scaler, 
         'not_norm_features': not_norm_features}

for col in norm_num_df_train.columns:
    train_df[col] = norm_num_df_train[col]
    test_df[col] = norm_num_df_test[col]

with open('scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_df.to_csv("train_processed_v4.csv", index=None)
test_df.to_csv("test_processed_v4.csv", index=None)