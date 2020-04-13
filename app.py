from flask import Flask
from flask_restful import Api, Resource, reqparse
import requests
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
from rq import Queue
from worker import conn

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
itoid = None
phrase_arr = None

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
            return {'message':'Please fecth phrases from chatbot agent API first.'}

def fetch_intents(itod, phase_arr):
    resp = requests.get('https://simple-chatbot-api.herokuapp.com/intents')
    intents = resp.json()['intents']
    itoid = []
    phrase_embs = []
    for intent in intents:
        for phrase in intent['phrases']:
            itoid.append(phrase['intent_id'])
            phrase_embs.append(model(phrase['value']).numpy())
    phrase_arr = np.vstack(phrase_embs)

class FetchIntents(Resource):

    def get(self):
        
        global itoid, phrase_arr
        # # get phrase from chatbot agent API
        # resp = requests.get('https://simple-chatbot-api.herokuapp.com/intents')
        # intents = resp.json()['intents']
        # itoid = []
        # phrase_embs = []
        # for intent in intents:
        #     for phrase in intent['phrases']:
        #         itoid.append(phrase['intent_id'])
        #         phrase_embs.append(model(phrase['value']).numpy())
        # phrase_arr = np.vstack(phrase_embs)
        q = Queue(connection=conn)
        q.enqueue(fetch_intents,itoid, phrase_arr)
        # self.fetch_intents(itoid, phrase_arr)
        
        return {'message':"Fetch phrases resource called!"}

def create_app():
    app = Flask(__name__)
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.secret_key = 'mick'
    return app

def create_api(app):
    api = Api(app)
    api.add_resource(FetchIntents, '/fetch_intents')
    api.add_resource(IntentClassifier, '/intent_classifier')

app = create_app()
create_api(app)

if __name__ == "__main__":
    app.run(port=5000, debug=True)