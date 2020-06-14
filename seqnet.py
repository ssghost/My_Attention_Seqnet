import os
import sys
import h5py
import datetime
import numpy as np 
import tensorflow as tf
import keras
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda, TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model, Model

import keras.backend as K
os.environ['KERAS_BACKEND'] = 'tensorflow'


class Seqnet:
    def __init__(self):
        self.in_tokens = None
        self.out_tokens = None
        self.scnt = 0
        self.inlen = 0
        self.outlen = 0
        self.inchars = 0
        self.outchars = 0
        self.model = None
        self.dec = None
        self.indict = None
        self.outdict = None
        self.callback = None

    def read(self,corpora,maxlen):
        if os.path.isfile(corpora):
            ext = os.path.splitext(corpora)[-1]
            if ext == '.txt':
                in_sentences, out_sentences = [],[]
                with open(corpora) as f:
                    lines = f.readlines()
                    self.scnt = int(len(lines)/2)
                    in_sentences.append(lines[::2])
                    out_sentences.append(lines[1::2])
                indict,outdict = [],[]
                for ch in in_sentences:
                    if ch not in in_chars:
                        indict.append()
                for ch in out_sentences:
                    if ch not in in_chars:
                        outdict.append()
                self.in_tokens = tf.zeros(shape=(self.scnt,maxlen[0],len(in_chars)))
                self.out_tokens = tf.zeros(shape=(self.scnt,maxlen[1],len(out_chars)))  
                for i in range(self.scnt):
                    for j,ch in enumerate(in_sentences[i]):
                        self.in_tokens[i,j,indict.index(ch)] = 1 
                    for j,ch in enumerate(out_sentences[i]):
                        self.out_tokens[i,j,outdict.index(ch)] = 1
                self.inlen = maxlen[0]
                self.outlen = maxlen[1]
                self.indict = indict
                self.outdict = outdict
                self.inchars = len(indict)
                self.outchars = len(outdict)
            else:
                print('File Format Dismatch.')
                sys.exit()
        else:
            print('File Not Found.')
            sys.exit()
    
    def one_step_attention(self,post,prev):
        X = RepeatVector(self.inlen)(prev)
        X = Concatenate(axis=-1)([X,post])

        X = TimeDistributed(Dense(self.outlen, activation = 'tanh'))(X)
        X = TimeDistributed(Dense(1, activation = 'relu'))(X)

        X = Activation(keras.activations.softmax(axis = 1), name = 'attention_weights')
        X = Dot(axes = 1)([X,post])

        return X
    
    def encoder(self, X):
        X = Bidirectional(LSTM(self.inchars*2, return_sequences=True), input_shape = (self.inlen,))(X)

        return X
    
    def decoder(self, dec, X):
        s,_,c = LSTM(self.outchars*4, return_state=True)(X, initial_state = dec)
        dec = [s,c]
        X = Dense(self.outchars, activation=keras.activations.softmax(axis = 1), kernel_initializer='glorot_uniform')(s)

        return dec, X
    
    def create_model(self):
        X = Input(shape=(self.inlen, self.inchars)) 
        dec = [Input(shape=(self.outchars*4,),name='dec_state_0'), Input(shape=(self.outchars*4,),name='dec_state_1')]
        X = self.encoder(X)

        outputs = []
        for t in range(self.outlen):
            X = self.one_step_attention(X, dec[0])
            dec, X = self.decoder(dec, X)
            outputs.append(X)
            
        outputs = keras.layers.Activation('softmax', name='attention_vec')(outputs)
        
        model = Model(inputs=[X,dec], outputs=np.array(outputs), name='SeqNet')

        return model

    def compile_model(self):
        self.model = self.create_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        print('Model Compiled.')
        self.model.summary()
        now = datetime.datetime.now()
        self.model.save('Model-'+now.strftime("%Y-%m-%d %H:%M:%S")+'.h5')

    def load_model(self, loadpath):
        if os.path.exists(loadpath):
            self.model = load_model(loadpath)
        else:
            print('Loadpath Not Found.')
            sys.exit()

    def train(self):
        self.dec = [tf.zeros((self.scnt, self.outchars*4)), tf.zeros((self.scnt, self.outchars*4))]
        outputs = self.out_tokens.swapaxes(0,1)
        self.model.fit([self.in_tokens,self.dec], outputs, epoch=100, batch_size=64, callbacks = [self.callback])

    def test(self, tpath, opath):
        if os.path.isfile(tpath):
            ext = os.path.splitext(tpath)[-1]
            if ext == '.txt':
                with open(tpath) as f:
                    lines = f.readlines()
                t_tokens = tf.zeros(shape=(len(lines),self.inlen,len(self.indict)))
                for i in range(len(lines)):
                    for j,ch in enumerate(lines[i]):
                        t_tokens[i,j,self.indict.index(ch)] = 1  
                result = self.model.predict(inputs=[t_tokens,self.dec])
                for i in range(len(result)):
                    result[i] = np.argmax(result[i], axis = -1)
                    result[i] = [self.outdict.index(int(j)) for j in result[i]]
                with open(opath,'w') as f:
                    for res in result:
                        f.writelines(res)
                        f.write('\n')
                print('Prediction Finished.')
            else:
                print('File Format Dismatch.')
                sys.exit()
        else:
            print('File Not Found.')
            sys.exit() 
        
    def callback(self):
        class acc_clip(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('acc') > 0.99):
                    self.model.stop_training = True
        self.callback = acc_clip()
        


                
