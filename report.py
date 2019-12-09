
import re
import os
import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt

from config import SR


def make_song_report(wav_x, wav_y, wav_y_pred, outdir, prefix='', is_save_audio=False):
    # -- plot waveform -- #
    # x
    plt.figure(figsize=(5, 2), dpi=100)
    librosa.display.waveplot(wav_x, sr=SR)
    plt.savefig(os.path.join(outdir, prefix + 'x'))
    plt.close()

    # y
    plt.figure(figsize=(5, 2), dpi=100)
    librosa.display.waveplot(wav_y, sr=SR)
    plt.savefig(os.path.join(outdir, prefix + 'y_ori'))
    plt.close()

    # y pred
    plt.figure(figsize=(5, 2), dpi=100)
    librosa.display.waveplot(wav_y_pred, sr=SR)
    plt.savefig(os.path.join(outdir, prefix + 'y_pred'))
    plt.close()

    # -- plot spec -- #
    # x
    plt.figure(dpi=100)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav_x)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.savefig(os.path.join(outdir, prefix + 'spec-x'))
    plt.close()

    # y
    plt.figure(dpi=100)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav_y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.savefig(os.path.join(outdir, prefix + 'spec-y_ori'))
    plt.close()

    # y pred
    plt.figure(dpi=100)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav_y_pred)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.savefig(os.path.join(outdir, prefix + 'spec-y_pred'))
    plt.close()


    # -- audio -- #
    if is_save_audio:
        librosa.output.write_wav(os.path.join(outdir, prefix + 'x.wav'), wav_x, sr=SR)
        librosa.output.write_wav(os.path.join(outdir, prefix + 'y.wav'), wav_y, sr=SR)
        librosa.output.write_wav(os.path.join(outdir, prefix + 'y_pred.wav'), wav_y_pred, sr=SR)


def make_loss_report(
        path_log,
        path_figure='loss.png',
        dpi=300, 
        x_range=None,
        y_range=None):
    
    # init
    list_loss = []
    list_loss_step = []
    list_valid = []
    list_valid_step = []
    training_time = ''
    
    # collect info
    with open(path_log) as f:
        for line in f:
            line = line.strip()
            if 'epoch' in line:
                loss = float(re.findall("loss: \d+\.\d+", line)[0][len('loss: '):])
                counter = int(re.findall("counter: \d+", line)[0][len('counter: '):])
                training_time = re.findall("time: \d+:\d+\:\d+.\d+", line)[0][len('time: '):]
                list_loss.append(loss)
                list_loss_step.append(counter)
            elif 'validation loss:' in line:
                loss = float(re.findall("validation loss: \d+\.\d+", line)[0][len('validation loss: '):])
                counter = int(re.findall("counter: \d+", line)[0][len('counter: '):])
                list_valid.append(loss)
                list_valid_step.append(counter)
            else:
                pass
            
    # plot
    fig = plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(list_loss_step, list_loss, label='train')
    plt.plot(list_valid_step, list_valid, label='valid')
    plt.legend(loc='upper right')
    if x_range:  
        plt.xlim(x_range[0], x_range[1])
    if y_range:  
        plt.xlim(y_range[0], y_range[1])
    txt = 'time: {}\ncounter: {}'.format(
            training_time,
            list_loss_step[-1]
        )
    fig.text(.5, -.05, txt, ha='center')
    plt.tight_layout()
    plt.savefig(path_figure)
    