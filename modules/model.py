import torch
import numpy as np
from typing import *
from logging import Logger
import pickle
import tensorflow as tf
from modules.base_model import BaseModel
import transformers
from transformers import TFAutoModel, AutoTokenizer

from modules.utils import *

AUTO = tf.data.experimental.AUTOTUNE

features_cat = ['Nonfiction', 'African American', 'Book Club', 'Realistic Fiction',
                'Retellings', 'Dark', 'Teen', 'Adventure', 'Paranormal', 'Humor',
                'Young Adult', 'Amazon', 'Asian Literature', 'Audiobook', 'Memoir',
                'Dystopia', 'Academic', 'Space', 'Detective', 'Novella', 'Horror', 'Race',
                'Novels', 'War', 'Adult Fiction', 'Fantasy', 'Cultural', 'Historical Romance',
                'Urban Fantasy', 'Fiction', 'Magical Realism', 'History', 'Contemporary Romance',
                'Erotica', 'Self Help', 'Food and Drink', 'Coming Of Age',
                'Books About Books', 'Autobiography', 'New Adult', 'Politics',
                'European Literature', 'Adult', 'Fairy Tales', 'Womens Fiction',
                'Health', 'Science Fiction Fantasy', 'Mythology', 'Sequential Art',
                'Romantic Suspense', 'Literature', 'Young Adult Fantasy', 'Childrens',
                'Short Stories', 'Science', 'Historical', 'Thriller', 'Crime', 'Family',
                'Sociology', 'Music', 'Biography', 'Psychology', 'Speculative Fiction',
                'Vampires', 'Magic', 'Middle Grade', 'Historical Fiction', 'Biography Memoir',
                'Writing', 'High School', 'Romance', 'Mystery Thriller', 'Mental Health',
                'Essays', 'Chick Lit', 'College', 'Sports', 'Witches', 'Graphic Novels',
                'Paranormal Romance', 'Religion', 'Contemporary', 'Queer', 'Business',
                'British Literature', 'Feminism', 'Mystery', 'World War II', 'Suspense',
                'Comics', 'Womens', 'Science Fiction', 'Drama', 'Literary Fiction',
                'Supernatural', 'Holiday', 'Environment', 'High Fantasy', 'Other_book_genre',
                'Penelope Douglas', 'Holly Black', 'Nora Roberts', 'Jennifer L. Armentrout',
                'Amy Harmon', 'Michael Connelly', 'Lucy Score', 'V.E. Schwab', 'Ilona Andrews',
                'Jay Kristoff', 'Rainbow Rowell', 'F. Scott Fitzgerald', 'Nalini Singh',
                'Christina Lauren', 'Seanan McGuire', 'Lauren Layne', 'Krista Ritchie',
                'Leigh Bardugo', 'Kelley Armstrong', 'Meghan Quinn', 'Lisa Jackson',
                'Jim Butcher', 'Helena Hunting', 'Melissa de la Cruz', 'Maxine Paetro',
                'J.R. Ward', 'Julia Quinn', 'Cassandra Clare', 'Danielle Steel', 'Louise Penny',
                'Rachel Caine', 'James Patterson', 'J.D. Robb', 'Stephen King', 'Marie Lu',
                'Fredrik Backman', 'John Grisham', 'Vi Keeland', 'Amie Kaufman', 'Meg Cabot',
                'Stuart Woods', 'Jewel E. Ann', 'Becca Ritchie', 'Danielle  Paige', 'Penelope Ward',
                'C.J. Box', 'L.J. Shen', 'Brandon Sanderson', 'Neil Gaiman', 'Other_book_authors']

features = ['book_pages', 'book_review_count', 'book_rating_count', 'image_width', 'image_heigth', 'image_size_product',
            'book_desc_len', 'ratio_review_rating', 'ratio_rating_review']


class Model(BaseModel):
    """
    Class to use bert sentence based model on inference
    :param logger: logger to use in model
    :param bert_model_path: path to saved fine-tuned bert model
    :param classification_model_path: path to saved fine-tuned classification_model
    """
    def __init__(self, logger: Logger, model_path: str, **kwargs):
        super().__init__(logger, **kwargs)
        print(model_path)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.tokenizer = AutoTokenizer.from_pretrained('albert-xxlarge-v2')
        print(self.model)
        self.logger.info("Models are loaded and ready to use.")

        self.logger.info("Loading cache...")

    def predict(self, book_title, book_image_url, book_desc, book_format, book_genre, book_authors,
                book_pages, book_review_count, book_rating_count):

        model_input = process_test_example(book_title, book_image_url, book_desc, book_genre,
                                           book_authors, book_format, book_pages, book_review_count, book_rating_count)

        x_test = self.dict_encode(list(model_input.book_desc.values))
        x_test.update({"features_cat": model_input[features_cat].values})
        x_test.update({"features_fea": model_input[features].values})


        test_dataset = (
            tf.data.Dataset
                .from_tensor_slices(x_test)
                .batch(1)
        )
        pred = self.model.predict(test_dataset, verbose=1)[0]

        return {"book_rating": pred}

    def dict_encode(self, texts, maxlen=512):
        enc_di = self.tokenizer.batch_encode_plus(
            texts,
            return_attention_mask=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            max_length=maxlen,
            truncation=True
        )

        return {
            "input_ids": np.array(enc_di['input_ids']),
            "attention_mask": np.array(enc_di['attention_mask'])
        }

    def regular_encode(self, texts, maxlen=512):
        enc_di = self.tokenizer.batch_encode_plus(
            texts,
            return_attention_mask=False,
            return_token_type_ids=False,
            pad_to_max_length=True,
            max_length=maxlen,
            truncation=True
        )

        return np.array(enc_di['input_ids'])
