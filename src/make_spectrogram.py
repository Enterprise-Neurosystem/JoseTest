#!/usr/bin/python3

import numpy as np
from scipy.io import wavfile
from scipy.fft import dct
import sys
import wave
import multiprocessing as mp



def processfile(params):
    params.setTid()
    with wave.open(params.fname,'r') as f:
        print(f.getparams())
        params.samplewidth = f.getsampwidth()
        params.totframes = f.getnframes()
        params.nchans = f.getnchannels()
        for c in range(params.nchans):
            params.data['ch%i'%c] = []
            params.filtdata['ch%i'%c] = []
        while f.tell()<min(params.totframes - params.nsamples*params.samplewidth*params.nchans,params.nchans*params.nsamples*params.nfolds*params.samplewidth):
            tmp = np.frombuffer(f.readframes(params.nsamples*params.samplewidth*params.nchans),dtype=params.dt)
            for c in range(params.nchans):
                params.data['ch%i'%c] += [np.log2(np.abs(dct(np.concatenate((tmp[c::params.nchans],np.flip(tmp[c::params.nchans],axis=0))),type=2,axis=0)[:1<<params.flim:2])/params.scale)]
                cepstrum = dct(np.concatenate((params.data['ch%i'%c][-1],np.flip(params.data['ch%i'%c][-1]))),axis=0,type=2)
                cepstrum[params.Poffset:params.Poffset+2*params.P:2] *= params.Pfilt
                cepstrum[params.Poffset+2*params.P:] *= 0.0
                cepstrum[:params.Poffset] *= 0.0
                back = dct(cepstrum,type=3,axis=0).real[:1<<(params.flim-1)]/params.scale
                back *= (back>0)
                params.filtdata['ch%i'%c] += [back]
                #params.filtdata['ch%i'%c] += [(1+np.tanh((back-params.thresh)/params.width))/2]
    for k in params.data.keys():
        oname = '%s.%s.sspect'%(params.fname,k)
        np.savetxt(oname,
                np.array(params.data[k]).T,
                fmt='%i',
                header='remember for %s, highest frequency is %i percent of the 96kHz (only showing 2**%i of 2**%i samples)'%(sys.argv[2],int(100*float(1<<params.flim)/float(params.nsamples)),1<<params.flim,np.log2(params.nsamples)) )
        print('finished writing:\t%s\tprocName:\t%s'%(oname,params.tid))
        
        oname = '%s.%s.sspect_filt'%(params.fname,k)
        np.savetxt(oname,
                np.array(params.filtdata[k]).T,
                fmt='%.1f',
                header='remember for %s, highest frequency is %i percent of the 96kHz (only showing 2**%i of 2**%i samples)'%(sys.argv[2],int(100*float(1<<params.flim)/float(params.nsamples)),1<<params.flim,np.log2(params.nsamples)) )
        print('finished writing:\t%s\tprocName:\t%s'%(oname,params.tid))
    return params

class Params:
    def __init__(self,fname,s):
        self.fname = fname
        self.setsubject(s)
        self.flim = 12
        self.tid = 'initState' 
        self.nchans = 1
        self.data = {}
        self.filtdata = {}
        self.dt = np.dtype(np.int16).newbyteorder('<')
        self.Poffset = 0
        self.thresh = 14
        self.width =1

    def initforsubject(self):
        if self.subject == 'bee':
            self.nsamples = 1<<14
            self.nfolds = 1<<12
            self.scale = 1<<10
        elif self.subject == 'todd':
            self.nsamples = 1<<12
            self.nfolds = 1<<14
            self.scale = 1<<10
        elif self.subject == 'server':
            self.nsamples = 1<<12
            self.nfolds = 1<<14
            self.scale = 1<<12
        return self

    def setsubject(self,s):
        self.subject = s
        self.initforsubject()
        return self
    def getsubject(self):
        return self.subject
    def setP(self,frac): #frac is the power of 2 in (1/2)**frac
        self.P = (1<<self.flim)>>frac
        return self
    def setFreqLim(self,f):
        self.flim = f
        return self
    def setPfilt(self):
        print(self.P)
        self.Pfilt = np.array([0.5 * (1 - np.cos(2*np.pi * x / self.P)) for x in range(self.P)])
        return self
    def setPoffset(self,p):
        self.Poffset = p
        return self
    def setTid(self):
        self.tid = '%s'%mp.current_process().name
        return self


def main():
    if len(sys.argv)<3:
        print("syntax:./src/make_spectrogram.py <'bee'|'server'|'todd'> <path/fnames>")
    
    paramslist = [Params(fname,sys.argv[1]) for fname in sys.argv[2:]]
    _ = [p.setFreqLim(12).setP(2).setPoffset(0).setPfilt() for p in paramslist]

    print('CPU cores:\t%i'%mp.cpu_count())
    _ = [print(p.fname) for p in paramslist]

    with mp.Pool(processes=len(paramslist)) as pool:
        pool.map(processfile,paramslist)
    
    return


    return

if __name__ == '__main__':
    main()



#### old method ######
    '''
    samplerate,data = wavfile.read(fname)
    print(samplerate)
    nsamples = 1<<14 
    nfolds = data.shape[0]>>10
    nfolds = 1<<12
    sz = nfolds*nsamples
    if len(data.shape)>1:
        ldata = data[:sz,0].reshape((nfolds,-1))
        rdata = data[:sz,1].reshape((nfolds,-1))
        RDATA = dct(np.column_stack((rdata,np.flip(rdata,axis=1))),type=2,axis=1)
        LDATA = dct(np.column_stack((ldata,np.flip(ldata,axis=1))),type=2,axis=1)
        np.savetxt('%s.rspect'%fname,np.abs(RDATA[:,:nsamples*2:2]).T)
        np.savetxt('%s.lspect'%fname,np.abs(LDATA[:,:nsamples*2:2]).T)
    else:
        sdata = data[:sz].reshape((nfolds,nsamples)).T
        #sdata = data[:sz].reshape((nsamples,nfolds))
        SDATA = dct(np.row_stack((sdata,np.flip(sdata,axis=0))),type=2,axis=0)
        np.savetxt('%s.sspect'%(fname),np.abs(SDATA[::2,:]),fmt='%.3f')
    print(samplerate)
    '''
