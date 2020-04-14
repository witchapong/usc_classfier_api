from flask import Flask
from flask_restful import Api, Resource, reqparse
import requests
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
from rq import Queue
from worker import conn
import boto3
import pickle
import os

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
itoid = None
phrase_arr = None
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')


class IntentClassifier(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('value',
        type=str,
        required=True,
        help="Sentence to send to chatbot agent cannot be empty."
        )

    @staticmethod
    def get_intent(sentence):
        # global phrase_arrs
        sent_vec = model(sentence).numpy()
        sim_score = sent_vec @ phrase_arr.T
        return int(pd.DataFrame({'intent_id':itoid, 'score':sim_score.squeeze()}).groupby('intent_id').sum().idxmax()['score'])

    def get(self):
        if phrase_arr is not None:
            payload = self.__class__.parser.parse_args()
            intent_id = self.get_intent(payload['value'])
            return {'intent_id':intent_id}
        else:
            return {'message':'Please fetch phrases from chatbot agent API first.'}


def put_on_s3():
    
    resp = requests.get('https://simple-chatbot-api.herokuapp.com/intents')
    intents = resp.json()['intents']
    itoid = []
    phrase_embs = []
    for intent in intents:
        for phrase in intent['phrases']:
            itoid.append(phrase['intent_id'])
            phrase_embs.append(model(phrase['value']).numpy())
    phrase_arr = np.vstack(phrase_embs)

    # dump to local
    pickle.dump(itoid, open('itoid.pkl','wb'))
    pickle.dump(phrase_arr, open('phrase_arr.pkl','wb'))

    # upload to S3
    s3_resource = boto3.resource('s3')
    s3_resource.Object(BUCKET_NAME, 'itoid.pkl').upload_file(Filename='itoid.pkl')
    s3_resource.Object(BUCKET_NAME, 'phrase_arr.pkl').upload_file(Filename='phrase_arr.pkl')
    print('uploaded files to S3 succesfully!')

class PutOnS3(Resource):

    def get(self):

        q = Queue(connection=conn)
        q.enqueue(put_on_s3)
        
        return {'message':"Fetch phrases resource called!"}


class PullFromS3(Resource):

    def get(self):
        global itoid, phrase_arr
        s3_resource = boto3.resource('s3')
        print('Downloading from S3')
        s3_resource.Object(BUCKET_NAME, 'itoid.pkl').download_file(f'itoid.pkl')
        s3_resource.Object(BUCKET_NAME, 'phrase_arr.pkl').download_file(f'phrase_arr.pkl')
        # load from local
        itoid = pickle.load(open('itoid.pkl','rb'))
        phrase_arr = pickle.load(open('phrase_arr.pkl','rb'))
        return {'message': "fecth result from S3 successfully!"}


def create_app():
    app = Flask(__name__)
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.secret_key = 'mick'
    return app

def create_api(app):
    api = Api(app)
    api.add_resource(PutOnS3, '/put_on_s3')
    api.add_resource(PullFromS3, '/pull_from_s3')
    api.add_resource(IntentClassifier, '/intent_classifier')

app = create_app()
create_api(app)

if __name__ == "__main__":
    app.run(port=5000, debug=True)