import datetime
import json
from argparse import ArgumentParser
import sys

from flask import Flask, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from pathlib import Path

# for debug mode
sys.path.insert(0, "/Users/ntr/Documents/tresh/zasrapi")

from modules.logging_utils import get_logger, check_if_none, ROOT_LOGGER_NAME
from modules.model import Model

parser = ArgumentParser()
parser.add_argument('--config', type=str, required=False,
                    default='modules/configs/config.json', help='path to config')

args = parser.parse_args()
config_path = args.config
logger = get_logger(name=ROOT_LOGGER_NAME,
                    console=True,
                    log_level="INFO",
                    propagate=False)

logger.info(f"Reading config from {Path(config_path).absolute()}")
with open(config_path) as con_file:
    config = json.load(con_file)
logger.info(f"Using config {config}")

logger.info(f"Loading model {config.get('model_name')}...")
model = Model(logger, **config)

# setting the api
app = Flask(__name__)
CORS(app)
api = Api(app, version=config.get("api_version", "0.0"), title='Int20h Final Submission')
ns1 = api.namespace('rating_model', description=config.get('model_name', 'Model'))

# response format
response = api.model('model_response', {
    'book_rating': fields.Float(required=True, description='neutral class probability'),
})


@ns1.route('/')
class TodoList(Resource):

    @ns1.doc('trigger_model')
    @ns1.param('book_title', _in='query')
    @ns1.param('book_image_url', _in='query')
    @ns1.param('book_desc', _in='query')
    @ns1.param('book_genre', _in='query')
    @ns1.param('book_authors', _in='query')
    @ns1.param('book_format', _in='query')
    @ns1.param('book_pages', _in='query')
    @ns1.param('book_review_count', _in='query')
    @ns1.param('book_rating_count', _in='query')
    @ns1.marshal_list_with(response)
    def get(self):
        start_time = datetime.datetime.now()
        book_title = request.args.get('book_title')
        book_image_url = request.args.get('book_image_url')
        book_desc = request.args.get('book_desc')
        book_format = request.args.get('book_format')
        book_genre = request.args.get('book_genre')
        book_authors = request.args.get('book_authors')
        book_pages = request.args.get('book_pages')
        book_review_count = int(request.args.get('book_review_count'))
        book_rating_count = int(request.args.get('book_rating_count'))

        print(book_title, book_image_url, book_desc, book_format, book_genre,
              book_authors, book_pages, book_review_count, book_rating_count)

        # inputs = model.process()
        result = model.predict(book_title, book_image_url, book_desc, book_format, book_genre,
                               book_authors, book_pages, book_review_count, book_rating_count)

        end_time = datetime.datetime.now()
        dif_time = str(end_time - start_time)

        logger.info(f'API; ModelOne Get response; difference: {dif_time}')

        return result


if __name__ == '__main__':
    app.run(debug=True, port=8001, host="0.0.0.0", threaded=True)
