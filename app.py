import streamlit as st
import spacy
from spacy import displacy
import streamlit.components.v1 as components
import SessionState
import time 

import os
import text_encoder
import six
import tensorflow as tf
import base64
import numpy as np
import copy
import re
from PIL import Image

import googleapiclient.discovery
from google.api_core.client_options import ClientOptions

from google.cloud import firestore


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vietai-research-8be1f340424d.json" # change for your GCP key
PROJECT = "vietai-research" # change for your GCP project
REGION = "asia-southeast1" # change for your GCP region (where your model is hosted)
MODEL = "translation_appendtag_envi_base_1000k"
ENVI_VERSION = 'envi_beam2_base1m'

vocab_file = 'vocab.subwords'


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s" % str((k, v)))
    # Subtly in PY2 vs PY3, map is not scriptable in py3. As a result,
    # map objects will fail with TypeError, unless converted to a list.
    if six.PY3 and isinstance(v, map):
      v = list(v)
    if (isinstance(v[0], six.integer_types) or
        np.issubdtype(type(v[0]), np.integer)):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example.SerializeToString()


def get_resource(version):
    # Create the ML Engine service object
    prefix = "{}-ml".format(REGION) if REGION else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(PROJECT, MODEL)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    predictor = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    return predictor, model_path


def check_mrs(content, i):
  is_mr = (i >= 2 and 
           content[i-2:i].lower() in ['mr', 'ms'] and
           (i < 3 or content[i-3] == ' '))
  is_mrs = (i >= 3 and 
            content[i-3:i].lower() == 'mrs' and 
            (i < 4 or content[i-4] == ' '))
  return is_mr or is_mrs


def check_ABB_mid(content, i):
  if i <= 0:
    return False
  if i >= len(content)-1:
    return False
  l, r = content[i-1], content[i+1]
  return l.isupper() and r.isupper()


def check_ABB_end(content, i):
  if i <= 0:
    return False
  l = content[i-1]
  return l.isupper()


def normalize(contents):
  # first step: replace special characters 
  check_list = ['\uFE16', '\uFE15', '\u0027','\u2018', '\u2019',
                '“', '”', '\u3164', '\u1160', 
                '\u0022', '\u201c', '\u201d', '"',
                '[', '\ufe47', '(', '\u208d',
                ']', '\ufe48', ')' , '\u208e', 
                '—', '_', '–', '&']
  alter_chars = ['?', '!', '&apos;', '&apos;', '&apos;',
                 '&quot;', '&quot;', '&quot;', '&quot;', 
                 '&quot;', '&quot;', '&quot;', '&quot;', 
                 '&#91;', '&#91;', '&#91;', '&#91;',
                 '&#93;', '&#93;', '&#93;', '&#93;', 
                 '-', '-', '-', '&amp;']

  replace_dict = dict(zip(check_list, alter_chars))

  new_contents = ''
  for i, char in enumerate(contents):
    if char == '&' and (contents[i:i+5] == '&amp;' or
                        contents[i:i+6] == '&quot;' or
                        contents[i:i+6] == '&apos;' or
                        contents[i:i+5] == '&#93;' or
                        contents[i:i+5] == '&#91;'):
      new_contents += char
      continue
    new_contents += replace_dict.get(char, char)
  contents = new_contents

  # second: add spaces
  check_sp_list = [',', '?', '!', '&apos;', '&amp;', '&quot;', '&#91;', 
                   '&#93;', '-', '/', '%', ':', '$', '#', '&', '*', ';', '=', '+', '$', '#', '@', '~', '>', '<']

  new_contents = ''
  i = 0
  while i < len(contents):
    char = contents[i]
    found = False
    for string in check_sp_list:
      if string == contents[i: i+len(string)]:
        new_contents += ' ' + string 
        if string != '&apos;':
          new_contents += ' '
        i += len(string)
        found = True
        break
    if not found:
      new_contents += char
      i += 1
  contents = new_contents

  # contents = contents.replace('.', ' . ')
  new_contents = ''
  for i, char in enumerate(contents):
    if char != '.':
      new_contents += char
      continue
    elif check_mrs(contents, i):
      # case 1: Mr. Mrs. Ms.
      new_contents += '. '
    elif check_ABB_mid(contents, i):
      # case 2: U[.]S.A.
      new_contents += '.'
    elif check_ABB_end(contents, i):
      # case 3: U.S.A[.]
      new_contents += '. '
    else:
      new_contents += ' . '

  contents = new_contents
  
  # third: remove not necessary spaces.
  new_contents = ''
  for char in contents:
    if new_contents and new_contents[-1] == ' ' and char == ' ':
      continue
    new_contents += char
  contents = new_contents
  
  return contents.strip()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache
def translate(from_txt):
  from_txt = from_txt.strip()
  input_ids = state.encoder.encode(from_txt) + [1]
  input_ids += [0] * (128 - len(input_ids))
  length = len(input_ids)
  byte_string = to_example({
      'inputs': list(np.array(input_ids, dtype=np.int32))
  })
  content = base64.b64encode(byte_string).decode('utf-8')

  # and the json to send is:
  input_data_json = {
      "signature_name": "serving_default",
      "instances": [{"b64": content}]
  } 


  request = state.model.predict(name=state.model_path, body=input_data_json)
  response = request.execute()
  
  if "error" in response:
      raise RuntimeError(response["error"])

  translated_text = ''
  for idx in response['predictions'][0]['outputs']:
      translated_text += state.vocab[idx][1:-1]
      
  to_text = translated_text.replace('_', ' ')
  to_text = to_text.replace('<EOS>', '').replace('<pad>', '')
  to_text = to_text.replace('& quot ;', '"')
  to_text = to_text.replace('& quot ;', '"')
  to_text = to_text.replace(' & apos ;', "'")
  to_text = to_text.replace(' , ', ', ')
  to_text = to_text.replace(' . ', ". ")
  to_text = to_text.split('\\')[0].strip()
  to_text = re.sub('\s+', ' ', to_text)
  return to_text


def write_ui():
  state.from_txt = st.text_area('Enter text to translate and click "Translate" ',
                            value=state.prompt,
                            height=100,
                            max_chars=600)
  
  button_value = st.button('Translate')
  state.ph0 = st.empty()

  if state.first_time :
    state.text_to_show = ''
  else:
    state.text_to_show = translate(state.from_txt)

  state.first_time = False
  
  state.user_edit = state.ph0.text_area(
                    'Translated text',
                    height=100,
                    value=state.text_to_show)

  if button_value or state.like:
    state.like = True
  
  different = normalize(state.user_edit) != normalize(state.text_to_show)
  
  state.col1, state.col2, state.col3, state.col4 = st.beta_columns([0.9, 0.25, 0.25, 3.0])
  with state.col2:
    state.ph2 = st.empty()

  if state.like:
    with state.col2:
      state.ph2 = st.empty()
      state.b2 = state.ph2.button('Yes')  
    with state.col3:
      state.ph3 = st.empty()
      state.b3 = state.ph3.button('No')   
    with state.col1:
      state.ph1 = st.markdown('Is this translation good ?')
  
    if state.b2:
      state.like = False
      state.ph1.empty()
      state.ph2.empty()
      state.ph3.empty()
      st.success('Thank you :)')
      state.db = firestore.Client.from_service_account_json("vietai-research-firebase-adminsdk.json")
        
      if state.direction_choice == "English to Vietnamese":
        state.db.collection(u"envi").add({
            u'from_text': state.from_txt,
            u'model_output': state.text_to_show,
            u'user_approve': True,
            u'time': time.time()
        })
      else:
        state.db.collection(u"vien").add({
            u'from_text': state.from_txt,
            u'model_output': state.text_to_show,
            u'user_approve': True,
            u'time': time.time()
        })

    elif state.b3:
      state.like = False
      state.ph1.empty()
      state.ph2.empty()
      state.ph3.empty()
      state.submit = True
  
  if state.submit:
    state.ph1.write('Make edit up here ⤴ and')

    if state.ph2.button('Submit'):
      if different:
         
        state.ph2.empty()
        state.ph1.empty()
        state.ph3.empty()
        
        state.submit = False
        # Save Users contribution:
        state.db = firestore.Client.from_service_account_json("vietai-research-firebase-adminsdk.json")
        
        if state.direction_choice == "English to Vietnamese":
          state.db.collection(u"envi").add({
              u'from_text': state.from_txt,
              u'model_output': state.text_to_show,
              u'user_translation': state.user_edit,
              u'time': time.time()
          })
        else:
          state.db.collection(u"vien").add({
              u'from_text': state.from_txt,
              u'model_output': state.text_to_show,
              u'user_translation': state.user_edit,
              u'time': time.time()
          })
        st.success("Your suggestion was recorded. Thank you :)")
        
        state.user_edit = state.ph0.text_area(
                    'Translated text',
                    height=100,
                    value=state.user_edit,
                    key=2)
  

st.set_page_config(
  page_title="Better translation for Vietnamese",
  layout='wide'
)

#Sidebar
st.sidebar.markdown('''
    <a href="https://vietai.org/" target="_blank" rel="noopener noreferrer">
        <img height="300" src="https://scontent.fsgn5-2.fna.fbcdn.net/v/t1.0-9/32905904_1247220778713646_5827247976073920512_o.png?_nc_cat=107&ccb=1-3&_nc_sid=85a577&efg=eyJpIjoidCJ9&_nc_ohc=ih_cV9hPIKIAX_Ew-Dd&tn=PRgOJlZwt8lThJU8&_nc_ht=scontent.fsgn5-2.fna&oh=b5d4c9d9769d3cd20c1414e2828bf3e6&oe=60884A55" />
    </a>''',
    unsafe_allow_html=True
)
st.sidebar.subheader("""Better translation for Vietnamese""")
st.sidebar.markdown("Authors: [Chinh Ngo](http://github.com/ntkchinh/) and [Trieu Trinh](http://github.com/thtrieu/).")
st.sidebar.markdown('Read more about this work [here](https://ntkchinh.github.io).')

# HtmlFile = open("test.html", 'r', encoding='utf-8')
# source_code = HtmlFile.read() 
# # # print(source_code)
# components.html(source_code, width=0, height=0)


#Main body
local_css("style.css")

directions = ['English to Vietnamese',
              'Vietnamese to English']

state = SessionState.get(like=False, submit=False, first_time=True, re_translate = False)

state.direction_choice = st.selectbox('Direction', directions)


@st.cache(allow_output_mutation=True)
def init(direction_choice):
  # print('rerunning {}'.format(state.direction_choice))
  if state.direction_choice == "English to Vietnamese":
    return (get_resource('envi_beam2_base1m'), 
            'Welcome to the best ever translation project for Vietnamese !')
  else:
    return (get_resource('vien_beam2_base1m'), 
            'Chào mừng bạn đến với dự án dịch tiếng Việt tốt nhất !')


state.encoder = text_encoder.SubwordTextEncoder(vocab_file)

with open(vocab_file, 'r') as f:
    state.vocab = f.read().split('\n')

(state.model, state.model_path), state.prompt = init(state.direction_choice)
write_ui()


