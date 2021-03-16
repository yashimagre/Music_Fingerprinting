# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:19:37 2019

@author: J.A.R.V.I.S
"""
import numpy as np
import pyaudio

p=pyaudio.PyAudio()
CHUNK=4096
RATE=44100

stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK)

for i in range(100):
    data1=np.fromstring(stream.read(CHUNK),dtype=np.int16)
for i in range(100):
    data2=np.fromstring(stream.read(CHUNK),dtype=np.int16)
   
stream.stop_stream()
stream.close()
p.terminate()