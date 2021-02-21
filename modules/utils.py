import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import cv2
import copy
from skimage import io
from matplotlib import pyplot as plt

import pickle

with open('modules/scaler.pickle', 'rb') as handle:
    scaler_dict = pickle.load(handle)


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


good_format = {'audible audio',
               'audio cd',
               'audiobook',
               'board book',
               'ebook',
               'hardcover',
               'kindle edition',
               'library binding',
               'mass market paperback',
               'paperback'}

good_authors = {'Amie Kaufman',
                'Amy Harmon',
                'Becca Ritchie',
                'Brandon Sanderson',
                'C.J. Box',
                'Cassandra Clare',
                'Christina Lauren',
                'Danielle  Paige',
                'Danielle Steel',
                'F. Scott Fitzgerald',
                'Fredrik Backman',
                'Helena Hunting',
                'Holly Black',
                'Ilona Andrews',
                'J.D. Robb',
                'J.R. Ward',
                'James Patterson',
                'Jay Kristoff',
                'Jennifer L. Armentrout',
                'Jewel E. Ann',
                'Jim Butcher',
                'John Grisham',
                'Julia Quinn',
                'Kelley Armstrong',
                'Krista Ritchie',
                'L.J. Shen',
                'Lauren Layne',
                'Leigh Bardugo',
                'Lisa Jackson',
                'Louise Penny',
                'Lucy Score',
                'Marie Lu',
                'Maxine Paetro',
                'Meg Cabot',
                'Meghan Quinn',
                'Melissa de la Cruz',
                'Michael Connelly',
                'Nalini Singh',
                'Neil Gaiman',
                'Nora Roberts',
                'Penelope Douglas',
                'Penelope Ward',
                'Rachel Caine',
                'Rainbow Rowell',
                'Seanan McGuire',
                'Stephen King',
                'Stuart Woods',
                'V.E. Schwab',
                'Vi Keeland'}

good_genres = {'Academic',
               'Adult',
               'Adult Fiction',
               'Adventure',
               'African American',
               'Amazon',
               'Asian Literature',
               'Audiobook',
               'Autobiography',
               'Biography',
               'Biography Memoir',
               'Book Club',
               'Books About Books',
               'British Literature',
               'Business',
               'Chick Lit',
               'Childrens',
               'College',
               'Comics',
               'Coming Of Age',
               'Contemporary',
               'Contemporary Romance',
               'Crime',
               'Cultural',
               'Dark',
               'Detective',
               'Drama',
               'Dystopia',
               'Environment',
               'Erotica',
               'Essays',
               'European Literature',
               'Fairy Tales',
               'Family',
               'Fantasy',
               'Feminism',
               'Fiction',
               'Food and Drink',
               'Graphic Novels',
               'Health',
               'High Fantasy',
               'High School',
               'Historical',
               'Historical Fiction',
               'Historical Romance',
               'History',
               'Holiday',
               'Horror',
               'Humor',
               'Literary Fiction',
               'Literature',
               'Magic',
               'Magical Realism',
               'Memoir',
               'Mental Health',
               'Middle Grade',
               'Music',
               'Mystery',
               'Mystery Thriller',
               'Mythology',
               'New Adult',
               'Nonfiction',
               'Novella',
               'Novels',
               'Paranormal',
               'Paranormal Romance',
               'Politics',
               'Psychology',
               'Queer',
               'Race',
               'Realistic Fiction',
               'Religion',
               'Retellings',
               'Romance',
               'Romantic Suspense',
               'Science',
               'Science Fiction',
               'Science Fiction Fantasy',
               'Self Help',
               'Sequential Art',
               'Short Stories',
               'Sociology',
               'Space',
               'Speculative Fiction',
               'Sports',
               'Supernatural',
               'Suspense',
               'Teen',
               'Thriller',
               'Urban Fantasy',
               'Vampires',
               'War',
               'Witches',
               'Womens',
               'Womens Fiction',
               'World War II',
               'Writing',
               'Young Adult',
               'Young Adult Fantasy'}


def one_hot_genres(df, good_genres, column="book_genre"):
    genres = df[column].apply(lambda x: np.array(x.split("|")) if type(x) == str else [])
    genres_dict = {genre: 0 for genre in good_genres}

    res_genres = []
    for row_genres in genres:
        row_genre_dict = copy.deepcopy(genres_dict)
        for g in row_genres:
            if g in good_genres:
                row_genre_dict[g] = 1
            else:
                row_genre_dict['Other_' + column] = 1
        res_genres.append(row_genre_dict)

    genre_df = pd.DataFrame(res_genres)

    for g in genre_df.columns:
        df[g] = genre_df[g].fillna(0)

    return df


def preprocessing(df):
    df['book_pages'] = df['book_pages'].apply(get_int)
    df['book_desc_len'] = df['book_desc'].apply(len)
    df['ratio_review_rating'] = df['book_review_count'] / df['book_rating_count']
    df['ratio_rating_review'] = df['book_rating_count'] / df['book_review_count']
    df['n1'] = df['book_image_url'].apply(number1_extraction)
    df['n2'] = df['book_image_url'].apply(number2_extraction)

    df = one_hot_genres(df, good_genres, column="book_genre")
    df = one_hot_genres(df, good_authors, column="book_authors")
    # df = one_hot_genres(df, good_format, column = "book_format")
    return df


def img_features(df, images):
    df['image_width'] = images.apply(lambda img: img.shape[0])
    df['image_heigth'] = images.apply(lambda img: img.shape[1])
    df['image_size_product'] = df['image_width'] * df['image_heigth']

    return df


def process_test_example(book_title="", book_image_url="", book_desc="", book_genre="", book_authors="", book_format="",
                         book_pages="", book_review_count=1, book_rating_count=1):
    test_examp = {'book_title': book_title,
                  'book_image_url': book_image_url,
                  'book_desc': book_desc,
                  'book_genre': book_genre,
                  'book_authors': book_authors,
                  'book_format': book_format,
                  'book_pages': book_pages,
                  'book_review_count': book_review_count,
                  'book_rating_count': book_rating_count}

    train_image = url_to_image(test_examp['book_image_url'])
    images = pd.Series([train_image])
    examples = [test_examp]
    df = pd.DataFrame(examples)
    df['book_format'] = df['book_format'].str.lower()

    not_norm_features = scaler_dict['not_norm_features']
    scaler = scaler_dict['scaler']

    df = preprocessing(df)
    df = img_features(df, images)

    df = df.replace([np.inf, -np.inf], 1)

    num_df_test = df[[col for col in df.columns if col not in not_norm_features]]
    num_df_test = num_df_test.fillna(0)
    columns = num_df_test.columns
    norm_num_df_test = scaler.transform(num_df_test)

    norm_num_df_test = pd.DataFrame(norm_num_df_test, columns=columns)

    for col in norm_num_df_test.columns:
        df[col] = norm_num_df_test[col]

    return df