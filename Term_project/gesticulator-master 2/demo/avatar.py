"""
Input
    Unity Voice Input으로 만든 음성 파일=audio1.wav
Output
    1. audio1.wav를 STT해서 얻은 text1.txt
    2. text1.txt와 audio1.wav를 Gesticulator에 넣어서 얻은 .bvh
"""



from argparse import ArgumentParser
import io
import os
import subprocess
from argparse import ArgumentParser
from google.cloud import texttospeech

### STT
# from __future__ import division

import shutil

import re
import sys

from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import speech
import pyaudio
# from six.moves import queue
from queue import Queue


import torch
import librosa
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from gesticulator.model.model import GesticulatorModel
from gesticulator.interface.gesture_predictor import GesturePredictor
from gesticulator.visualization.motion_visualizer.generate_videos import visualize



# ---------------------------------------------------------------------------------------------------------


'''Gesticulator'''
def main(args, my_response):
    
    # 0. Check feature type based on the model
    feature_type, audio_dim = check_feature_type(args.model_file)

    # 1. Load the model
    model = GesticulatorModel.load_from_checkpoint(
        args.model_file, inference_mode=True)
    # This interface is a wrapper around the model for predicting new gestures conveniently
    gp = GesturePredictor(model, feature_type)

    # 2. Predict the gestures with the loaded model
    print(f"my_response = {my_response}")
    motion = gp.predict_gestures(args.audio, my_response)

    # 3. Visualize the results
    motion_length_sec = int(motion.shape[1] / 20)

    visualize(motion.detach(), "avatar.bvh", "temp.npy", "temp.mp4", 
              start_t = 0, end_t = motion_length_sec, 
              data_pipe_dir = 'C:/Users/woduc/Desktop/inde_unity/gesticulator-master/gesticulator/utils/data_pipe.sav')


    # Add the audio to the video
    command = f"ffmpeg -y -i {args.audio} -i temp.mp4 -c:v libx264 -c:a libvorbis -loglevel quiet -shortest {args.video_out}"
    subprocess.call(command.split())

    print("/nGenerated video:", args.video_out)
    
    # Remove temporary files
    for ext in [ "npy", "mp4"]:
        os.remove("temp." + ext)

def check_feature_type(model_file):
    """
    Return the audio feature type and the corresponding dimensionality
    after inferring it from the given model file.
    """
    params = torch.load(model_file, map_location=torch.device('cpu'))

    # audio feature dim. + text feature dim.
    audio_plus_text_dim = params['state_dict']['encode_speech.0.weight'].shape[1]

    # This is a bit hacky, but we can rely on the fact that 
    # BERT has 768-dimensional vectors
    # We add 5 extra features on top of that in both cases.
    text_dim = 768 + 5

    audio_dim = audio_plus_text_dim - text_dim

    if audio_dim == 4:
        feature_type = "Pros"
    elif audio_dim == 64:
        feature_type = "Spectro"
    elif audio_dim == 68:
        feature_type = "Spectro+Pros"
    elif audio_dim == 26:
        feature_type = "MFCC"
    elif audio_dim == 30:
        feature_type = "MFCC+Pros"
    else:
        print("Error: Unknown audio feature type of dimension", audio_dim)
        exit(-1)

    return feature_type, audio_dim


def truncate_audio(input_path, target_duration_sec):
    """
    Load the given audio file and truncate it to 'target_duration_sec' seconds.
    The truncated file is saved in the same folder as the input.
    """
    audio, sr = librosa.load(input_path, duration = int(target_duration_sec))
    output_path = input_path.replace('.wav', f'_{target_duration_sec}s.wav')

    librosa.output.write_wav(output_path, audio, sr)

    return output_path

def parse_args():
    # path = 'C:/Users/SOGANG/Documents/developer/I/gesticulator-master/demo/'
    parser = ArgumentParser()
    parser.add_argument('--audio', type=str, default="C:/Users/woduc/Desktop/inde_unity/Assets/Test1.wav", help="path to the input speech recording")
    # parser.add_argument('--text', type=str, default="input/jeremy_howard.json",
    #                     help="one of the following: "
    #                          "1) path to a time-annotated JSON transcription (this is what the model was trained with) "
    #                          "2) path to a plaintext transcription, or " 
    #                          "3) the text transcription itself (as a string)")
    parser.add_argument('--video_out', '-video', type=str, default="output/generated_motion.mp4",
                        help="the path where the generated video will be saved.")
    parser.add_argument('--model_file', '-model', type=str, default="models/default.ckpt",
                        help="path to a pretrained model checkpoint")
    parser.add_argument('--mean_pose_file', '-mean_pose', type=str, default="../gesticulator/utils/mean_pose.npy",
                        help="path to the mean pose in the dataset (saved as a .npy file)")
    
    return parser.parse_args()

# ----------------------------------------------------------------------------

'''STT - .wav version'''
def stt():
    # Instantiates a client
    # [START migration_client]
    client = speech.SpeechClient()
    # [END migration_client]

    # The name of the audio file to transcribe
    file_name = "C:/Users/woduc/Desktop/inde_unity/Assets/Test1.wav"

    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='en-US')

    # print(f"config = {config}")
    # print(f"audio = {audio}")
    # Detects speech in the audio file
    response = client.recognize(config=config, audio=audio)

    text = []
    for result in response.results:
    #     print('Transcript: {}'.format(result.alternatives[0].transcript))
        text.append(result.alternatives[0].transcript)
        print(f"audio1.wav = {text}")
    # [END speech_quickstart]
    return text



# Run all

if __name__ == "__main__":
    my_message_list = []
    
    """ [STT] audio.wav -> text """
    input_text = stt()[0]
    print(f"input text = {input_text}")
    input_text = input_text + '.' # Gesticulator에서 문장의 마지막에 . 으로 끝나야 input text가 제대로 들어감. STT만하면 문장의 마지막에 . 이 없기 때문에 추가해줘야함.
    
    # Create text1.txt file
    f = open('text1.txt','w')
    f.write(input_text)
    f.close()
    
    
    """ [Gesticulator] response text, response audio.wav -> gesture bvh """
    args = parse_args()
    main(args, input_text) # args = audio, my_response=input_text



