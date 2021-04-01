import streamlit as st
import os
import text_encoder
import six
import tensorflow as tf
import base64
import numpy as np

import googleapiclient.discovery
from google.api_core.client_options import ClientOptions


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vietai-research-8be1f340424d.json" # change for your GCP key
PROJECT = "vietai-research" # change for your GCP project
REGION = "asia-southeast1" # change for your GCP region (where your model is hosted)
MODEL = "translation_appendtag_envi_base_1000k"
ENVI_VERSION = 'envi_beam2_base1m'

vocab_file = 'vocab.subwords'

encoder = text_encoder.SubwordTextEncoder(vocab_file)

with open(vocab_file, 'r') as f:
    vocab = f.read().split('\n')


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


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


def write_ui():
  from_txt = st.text_input('Enter text to translate and hit Enter',
                            value=prompt,
                            max_chars=600)
  if not from_txt:
      return

    # from_txt = normalize(from_txt)

  from_txt = from_txt.strip()

  input_ids = encoder.encode(from_txt) + [1]
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


  request = model.predict(name=model_path, body=input_data_json)
  response = request.execute()
  
  if "error" in response:
      raise RuntimeError(response["error"])

  translated_text = ''
  for idx in response['predictions'][0]['outputs']:
      translated_text += vocab[idx][1:-1]
      
  to_text = translated_text.replace('_', ' ')
  to_text = to_text.replace('<EOS>', '').replace('<pad>', '')
  to_text = to_text.replace('& quot ;', '"')
  to_text = to_text.replace('& quot ;', '"')
  to_text = to_text.replace(' & apos ;', "'")
  to_text = to_text.replace(' , ', ', ')
  to_text = to_text.replace(' . ', ". ")
  to_text = to_text.split('\\')[0].strip()
  
  output_text = st.text_area('Translated text',
                              height=100,
                              value=to_text)


st.set_page_config(
  page_title="Better translation for Vietnamese",
  layout='wide'
  )
#Sidebar
st.sidebar.image(
    "https://scontent.fsgn5-2.fna.fbcdn.net/v/t1.0-9/32905904_1247220778713646_5827247976073920512_o.png?_nc_cat=107&ccb=1-3&_nc_sid=85a577&efg=eyJpIjoidCJ9&_nc_ohc=ih_cV9hPIKIAX_Ew-Dd&tn=PRgOJlZwt8lThJU8&_nc_ht=scontent.fsgn5-2.fna&oh=b5d4c9d9769d3cd20c1414e2828bf3e6&oe=60884A55",
    width=250, height=150,
)

#Main body
local_css("style.css")
st.title("""
Better translation for Vietnamese
Read more about our work [here](https://ntkchinh.github.io).
""")

directions = ['English to Vietnamese',
              'Vietnamese to English']

direction_choice = st.selectbox('Direction', directions)

if direction_choice == "English to Vietnamese":
    model, model_path = get_resource('envi_beam2_base1m')
    prompt = 'Welcome to the best ever translation project for Vietnamese !'
else:
    model, model_path = get_resource('vien_beam2_base1m')
    prompt = 'Chào mừng đến với dự án dịch tiếng Việt tốt nhất  !'


write_ui()






