"""Constant-time ZIP member lookup avoids quadratic NPZ name scans. Read-only, no pickle."""
from zipfile import ZipFile
import numpy as np

class Arrays:
    def __init__(self,path):self.zip=ZipFile(path);self.files=[n[:-4] for n in self.zip.namelist() if n.endswith('.npy')];self.cache={}
    def __getitem__(self,key):
        if key not in self.cache:
            with self.zip.open(key+'.npy') as f:self.cache[key]=np.lib.format.read_array(f,allow_pickle=False)
        return self.cache[key]
    def close(self):self.zip.close()
