import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# পেজ কনফিগারেশন
st.set_page_config(page_title="Image Captioning App", page_icon="🖼️", layout="centered")

st.title("🖼️ AI Image Captioning App")
st.write("একটি ছবি আপলোড করুন এবং আমাদের ডিপ লার্নিং মডেল ছবিটির ক্যাপশন তৈরি করবে!")

# হাইপারপ্যারামিটার
embedding_dim = 256
units = 512
top_freq_words = 5000
vocab_size = top_freq_words + 1
max_length = 35 # আপনার নোটবুকের ম্যাক্স লেংথ অনুযায়ী
attention_features_shape = 64

# ১. মডেল আর্কিটেকচার ডিফাইন করা
class Encoder(Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embedding_dim)
    def call(self, features):
        features = self.dense(features)
        features = tf.nn.relu(features)
        return features

class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.units = units
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = Attention_model(self.units)
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units)
        self.d2 = tf.keras.layers.Dense(vocab_size)
    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        embed = self.embed(x)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
        output, state = self.gru(embed)
        x = self.d1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.d2(x)
        return x, state, attention_weights
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

@st.cache_resource
def load_assets():
    # Tokenizer লোড করা
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    # InceptionV3 ফিচার এক্সট্র্যাক্টর লোড করা
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
    # এনকোডার ও ডিকোডার ইনিশিয়ালাইজ করা
    encoder = Encoder(embedding_dim)
    decoder = Decoder(embedding_dim, units, vocab_size)
    
    # চেকপয়েন্ট রিস্টোর করা
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints/train", max_to_keep=5)
    
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Checkpoint restored successfully!")
        
    return tokenizer, image_features_extract_model, encoder, decoder

tokenizer, image_features_extract_model, encoder, decoder = load_assets()

# ইমেজ প্রিপ্রসেস ফাংশন
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

# ইভালুয়েট বা ক্যাপশন জেনারেট করার ফাংশন
def evaluate(image_path):
    temp_input = tf.expand_dims(preprocess_image(image_path), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    hidden = decoder.init_state(batch_size=1)
    
    result = []
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        
        word = tokenizer.index_word.get(predicted_id, '<unk>')
        if word == '<end>':
            break
        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return ' '.join(result)

# স্ট্রিমলিট ইউজার ইন্টারফেস
uploaded_file = st.file_uploader("একটি ছবি নির্বাচন করুন (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ছবিটি সেভ করে রাখা প্রসেসিংয়ের জন্য
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # ছবি ডিসপ্লে করা
    st.image(uploaded_file, caption='আপলোড করা ছবি', use_column_width=True)
    
    if st.button('🚀 ক্যাপশন জেনারেট করুন'):
        with st.spinner('মডেল ক্যাপশন তৈরি করছে... একটু অপেক্ষা করুন!'):
            caption = evaluate(temp_file_path)
            st.success("**জেনারেটেড ক্যাপশন:**")
            st.markdown(f"### *{caption.capitalize()}*")