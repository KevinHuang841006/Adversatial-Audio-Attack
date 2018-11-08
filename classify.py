## python classify.py XXX.wav

import numpy as np
import tensorflow as tf
import argparse

import scipy.io.wavfile as wav

import struct
import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")

#import DeepSpeech
# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]
class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v
tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = tmp

# More monkey-patching, to stop the training coordinator setup
#DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
#DeepSpeech.TrainingCoordinator.start = lambda x: None


from util.text import ctc_label_dense_to_sparse
#from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

def compute_mfcc(audio, **kwargs):
    """
    Compute the MFCC for a given audio waveform. This is
    identical to how DeepSpeech does it, but does it all in
    TensorFlow so that we can differentiate through it.
    """

    batch_size, size = audio.get_shape().as_list()
    audio = tf.cast(audio, tf.float32)

    # 1. Pre-emphasizer, a high-pass filter
    audio = tf.concat((audio[:, :1], audio[:, 1:] - 0.97*audio[:, :-1], np.zeros((batch_size,1000),dtype=np.float32)), 1)

    # 2. windowing into frames of 320 samples, overlapping
    windowed = tf.stack([audio[:, i:i+400] for i in range(0,size-320,160)],1)

    # 3. Take the FFT to convert to frequency space
    ffted = tf.spectral.rfft(windowed, [512])
    ffted = 1.0 / 512 * tf.square(tf.abs(ffted))

    # 4. Compute the Mel windowing of the FFT
    energy = tf.reduce_sum(ffted,axis=2)+1e-30
    filters = np.load("filterbanks.npy").T
    feat = tf.matmul(ffted, np.array([filters]*batch_size,dtype=np.float32))+1e-30

    # 5. Take the DCT again, because why not
    feat = tf.log(feat)
    feat = tf.spectral.dct(feat, type=2, norm='ortho')[:,:,:26]

    # 6. Amplify high frequencies for some reason
    _,nframes,ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff)
    lift = 1 + (22/2.)*np.sin(np.pi*n/22)
    feat = lift*feat
    width = feat.get_shape().as_list()[1]

    # 7. And now stick the energy next to the features
    feat = tf.concat((tf.reshape(tf.log(energy),(-1,width,1)), feat[:, :, 1:]), axis=2)
    
    return feat

def get_mfcc(new_input, length):    
    # We need to init DeepSpeech the first time we're called
    

    batch_size = new_input.get_shape()[0]
    new_input_to_mfcc = compute_mfcc(new_input)[:, ::2]
    empty_context = np.zeros((batch_size, 9, 26), dtype=np.float32)
    new_input_to_mfcc = compute_mfcc(new_input)[:, ::2]
    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)
    return features
    
def get_logits(new_input, length, first=[]):
    """
    Compute the logits for a given waveform.

    First, preprocess with the TF version of MFC above,
    and then call DeepSpeech on the features.
    """

    # We need to init DeepSpeech the first time we're called
    if first == []:
        first.append(False)
        # Okay, so this is ugly again.
        # We just want it to not crash.
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()

    batch_size = new_input.get_shape()[0]

    # 1. Compute the MFCCs for the input audio
    # (this is differentable with our implementation above)
    empty_context = np.zeros((batch_size, 9, 26), dtype=np.float32)
    new_input_to_mfcc = compute_mfcc(new_input)[:, ::2]
    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)

    # 2. We get to see 9 frames at a time to make our decision,
    # so concatenate them together.
    features = tf.reshape(features, [new_input.get_shape()[0], -1])
    features = tf.stack([features[:, i:i+19*26] for i in range(0,features.shape[1]-19*26+1,26)],1)
    features = tf.reshape(features, [batch_size, -1, 19*26])

    # 3. Whiten the data
    mean, var = tf.nn.moments(features, axes=[0,1,2])
    features = (features-mean)/(var**.5)

    # 4. Finally we process it with DeepSpeech
    logits = DeepSpeech.BiRNN(features, length, [0]*10)

    return logits

from python_speech_features import mfcc
from python_speech_features import logfbank
    
def main():
    #with tf.Session() as sess:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    
    
    sess = tf.Session(config=config)
    print("stuck")
    for i in range(1,len(sys.argv)):
            
            
            # 2 type of audio file

        if sys.argv[i].split(".")[-1] == 'wav':
            _, audio = wav.read(sys.argv[i])
        else:
            raise Exception("Unknown file format")
				
        N = len(audio)
        new_input = tf.placeholder(tf.float32, [1, N])
        lengths = tf.placeholder(tf.int32, [1])
			
		# make tensor can have same name variable_scope
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logits = get_logits(new_input, lengths)
        
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logs = get_mfcc(new_input, lengths)
        
        if i == 1:
            saver = tf.train.Saver()
            saver.restore(sess, "models/session_dump")
        
        
        (rate,sig) = wav.read(sys.argv[i])
        mfcc_feat = mfcc(sig,rate)
        fbank_feat = logfbank(sig,rate)
        print("rate: ",rate)
        print("sig:")
        print(len(sig))
        print("mfcc: ")
        print(fbank_feat[1:3,:])
        print("mfcc~: ")
        fbank_feat = np.array(fbank_feat)
        print(fbank_feat.shape)
        
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=True, beam_width=500)
        print("audio len: ", len(audio))
        length = (len(audio)-1)//320
        print("output len: ", length)
        l = len(audio)
        r = sess.run(decoded, {new_input: [audio],
                                   lengths: [length]})
        x_example = sess.run( logs , {new_input: [audio], lengths: [length]})
        print(x_example)
        x_example = np.array(x_example)
        print(x_example.shape)
        shows = tf.gradients(logits, new_input)
        print(sess.run(shows, {new_input: [audio], lengths: [length]}))                           
                                   
        if len(sys.argv[i]) > 2:
            print(sys.argv[i])
        print(" ".join([toks[x] for x in r[0].values]))
        print("-----------------------------------")
        
main()
