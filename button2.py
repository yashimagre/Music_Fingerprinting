# -*- coding: utf-8 -*-


"""
Created on Thu Jul  4 12:34:22 2019

@author: Yashi
"""
import tkinter
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import sigproc
from sklearn.model_selection import train_test_split

window = tkinter.Tk()
window.title("Lizzy")
label = tkinter.Label(window, text = "Welcome to Lizzy(songs)").pack()

def calculate_nfft(samplerate, winlen):
    #Calculates the FFT size as a power of two greater than or equal to the number of samples in a single window length.
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:np.ones((x,))):
    #Compute MFCC features from an audio signal.
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          winfunc=lambda x:np.ones((x,))):
    #Compute Mel-filterbank energy features from an audio signal.
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    energy = np.sum(pspec,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy

def logfbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
             nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
             winfunc=lambda x:np.ones((x,))):
    #Compute log Mel-filterbank energy features from an audio signal.
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    return np.log(feat)

def ssc(signal,samplerate=16000,winlen=0.025,winstep=0.01,
        nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
        winfunc=lambda x:np.ones((x,))):
    #Compute Spectral Subband Centroid features from an audio signal.
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    pspec = np.where(pspec == 0,np.finfo(float).eps,pspec) # if things are all zeros we get problems

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    R = np.tile(np.linspace(1,samplerate/2,np.size(pspec,1)),(np.size(pspec,0),1))

    return np.dot(pspec*R,fb.T) / feat

def hz2mel(hz):
    #Convert a value in Hertz to Mels
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    #Convert a value in Mels to Hertz
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    #Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra, L=22):
    #Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the magnitude of the high frequency DCT coeffs.
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def delta(feat, N):
    #Compute delta features from a feature vector sequence.
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat
class Config:
    def __init__(self,mode='conv',nfilt=26,nfeat=13,nfft=512,rate=16000):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft=nfft
        self.rate=rate
        self.step=int(rate/10)
df = pd.read_csv('data.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('samples/' +f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes=list(np.unique(df.label))
class_dist=df.groupby(['label'])['length'].mean()

n_samples=5*int(df['length'].sum())
prob_dist=class_dist/class_dist.sum()
choices=np.random.choice(class_dist.index,p=prob_dist)

def build_rand_feat():
    x=[]
    y=[]
    
    _min,_max=float('inf'),-float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class=np.random.choice(class_dist.index,p=prob_dist)
        file=np.random.choice(df[df.label==rand_class].index)
        rate,wav=wavfile.read('samples/'+file)
        label=df.at[file,'label']
        if wav.shape[0]!=0 and wav.shape[0]>config.step:
            rand_index=np.random.randint(0,wav.shape[0]-config.step)
            sample=wav[rand_index:rand_index+config.step]
            x_sample=mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,nfft=config.nfft).T
            _min=min(np.amin(x_sample),_min)
            _max=max(np.amax(x_sample),_max)
            x.append(x_sample )
            y.append(classes.index(label))
    x,y=np.array(x),np.array(y)
    x=(x-_min)/(_max-_min)
    if config.mode=='conv':
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    elif config.mode=='time':
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2])
    y=to_categorical(y,num_classes=34)
    
    return x,y
def get_conv_model():
    model=Sequential()
    model.add(Conv2D(34,(3,3),activation='relu',strides=(1,1),padding='same',input_shape=input_shape))
    model.add(Conv2D(32,(3,3),activation='relu',strides=(1,1),padding='same'))
    model.add(Conv2D(64,(3,3),activation='relu',strides=(1,1),padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',strides=(1,1),padding='same'))
    model.add(Conv2D(256,(3,3),activation='relu',strides=(1,1),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same'))
    
    
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(128,activation='relu'))
    
    model.add(Dense(34,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
config=Config(mode='conv')
if config.mode=='conv':
    X,Y=build_rand_feat()
    y_flat=np.argmax(Y,axis=1)
    input_shape=(X.shape[1],X.shape[2],1)
    model=get_conv_model()
    



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()


#class_weight=compute_class_weight('balanced',np.unique(y_flat),y_flat)
model.fit(X_train,Y_train,epochs=10,batch_size=32,shuffle=True)
score=model.evaluate(X_test,Y_test)
q=np.argmax(Y_test,axis=1)
y=model.predict(X_test)
w=np.argmax(y,axis=1)
cm=confusion_matrix(q,w)
print(cm)
print(score[1])
def start():
    print('function called')
    tkinter.Label(window, text = " Let's get Started..!!").pack()
    import pyaudio
    import numpy as np

    CHUNK = 4096 # number of data points to read at a time
    RATE = 16000 # time resolution of the recording device (Hz)

    p=pyaudio.PyAudio() # start the PyAudio class
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK) #uses default input device
    data1=[]
    
    
# create a numpy array holding a single read of audio data
    for i in range(10): #to it a few times just to see
        data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
        data1.append(data)
        print(data)
    data1=np.array(data1)
    print(data1.shape)    

    x=[]
    y=[]
    _min,_max=float('inf'),-float('inf')
    df = pd.read_csv('data.csv')
    df.set_index('fname', inplace=True)

    classes=list(np.unique(df.label))
    class Config:
        def __init__(self,mode='conv',nfilt=26,nfeat=13,nfft=512,rate=16000):
            self.mode=mode
            self.nfilt=nfilt
            self.nfeat=nfeat
            self.nfft=nfft
            self.rate=rate
            self.step=int(rate/10)
    config=Config(mode='conv')
    for i in range(10):
        if len(data1[i])!=0 and len(data1[i])>config.step:
            rand_index=np.random.randint(0,len(data1[i])-config.step)
            sample=data1[i][rand_index:rand_index+config.step]
            x_sample=mfcc(sample,samplerate=RATE,numcep=config.nfeat,nfilt=config.nfilt,nfft=config.nfft).T
            _min=min(np.amin(x_sample),_min)
            _max=max(np.amax(x_sample),_max)
            x.append(x_sample)
    x=np.array(x)
    x=(x-_min)/(_max-_min)
    print(x.shape)
    if config.mode=='conv':
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    print(x.shape)
    
    p.terminate()
    stream.stop_stream()
    stream.close()
    y=model.predict(x)
    g=list(np.argmax(y,axis=1))
    res=max(set(g),key=g.count)
    print(classes[res])
    
   
   

    
def stop():
    tkinter.Label(window, text = "Stopped").pack()
    #tkinter.Label(window, text = y).pack()
    window.destroy()
startButton = tkinter.Button(window, height=2, width=20, text ="Start", command = start)
stopButton = tkinter.Button(window, height=2, width=20, text ="Stop", command = stop)
startButton.pack()
stopButton.pack()
window.mainloop()

