import requests
import numpy as np
import pickle
import boto3
import os
import tensorflow_hub as hub
import tensorflow_text
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

def put_on_s3():
    
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
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