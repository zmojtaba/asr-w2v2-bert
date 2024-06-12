import pandas as pd
import torchaudio
import numpy as np
file_path = "/home/hoosh-2/project/sedava/datasets/data/"

def read_csv(file_name):
    df = pd.read_csv( file_path + "/subtitle.csv" )
    return df

def read_audio(file_name):
    waveform, sample_rate   = torchaudio.load(file_name)
    return waveform,sample_rate

def create_csv_with_length(df):
    df['length'] = np.nan  
    for i in range( len(df) ):
        print(i)
        waveform, sample_rate =  read_audio( file_path + "chunked_audio/" +  df['file_name'][i] + ".mp3" )

        if len(waveform)>1:
            waveform = waveform[0]
        df.at[i, 'length'] = np.float16( len(waveform) / sample_rate )
    return df


print()
df = read_csv(file_path)
new_df = create_csv_with_length(df)

new_df.to_csv( file_path + "subtitle_with_length.csv" )