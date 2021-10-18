import random
import librosa
import itertools
import numpy as np
from tqdm.auto import tqdm


def compute_mfcc(waveforms, n_mfcc, sr=16000):
    """
    Parameters
    ----------
    data: List[np.ndarray]
        A list of waveform np.array.
    n_mfcc: int
        The number of cepstrum to return.
    sr: int
        Target sampling rate.
    """
    mfccs = []
    for wav in tqdm(waveforms, desc='MFCC', leave=False):
        # feature = python_speech_features.mfcc(wav, 
        #                                       samplerate=sr, 
        #                                       numcep=n_mfcc, 
        #                                       nfft=int(sr*0.025))
        feature = librosa.feature.mfcc(wav, sr=sr, n_mfcc=n_mfcc)
        mfccs.append(feature)
    return np.array(mfccs)


def contiguous_slice(sequence, windows, num_samples, seed=914):
    random.seed(seed)
    lst = range(len(sequence))
    list_size = len(lst)
    indexes = [lst[i:i+windows] for i in range(list_size-windows+1)]
    indexes = random.choices(indexes, k=num_samples)
    sub_lst = [sequence[min(idx):max(idx)+1] for idx in indexes]
    return sub_lst


def pad_sequences(sequences, maxlen=None, padding='zero', batch=False):
    if not batch:
        if padding == 'zero':
            seq = np.array(zero_padding(sequences, maxlen, filler=0))
        if padding == "repeat":
            seq = np.array(repeat_padding(sequences, maxlen))
        return seq
    pad_seq = []
    for seq in sequences:
        if padding == 'zero':
            seq = np.array(zero_padding(seq, maxlen, filler=0))
        if padding == "repeat":
            seq = np.array(repeat_padding(seq, maxlen))
        pad_seq.append(seq)
    return np.array(pad_seq)


def repeat_padding(seq, size):
    """
    Parameters
    ----------
    seq: np.array
    
    Returns
    -------
    list

    References
    ----------
    1. https://stackoverflow.com/a/60972703
    """
    src = seq
    trg = [0] * size
    data = [src, trg]
    m = len(max(data, key=len))
    r = list(itertools.starmap(np.resize, ((e, m) for e in data)))
    return r[0][:size]


def zero_padding(seq, size, filler=0):
    """
    Parameters
    ----------
    seq: np.array
        The sequence to be padded.
    size: int
        The size of the output sequence.
    filler: float or int
        Pads with a constant value.
        
    Returns
    -------
    list

    References
    ----------
    1. https://stackoverflow.com/a/30475648
    """
    return list(itertools.islice(itertools.chain(seq, itertools.repeat(filler)), size))

