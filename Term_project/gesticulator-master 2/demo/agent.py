"""
Input
    Avatar에서 생성된 텍스트파일=text1.txt
Output
    1. DialoGPT를 거쳐 생성된 텍스트(=text2.txt)를 TTS해서 얻은 audio2.wav
    2. text2.txt와 audio2.wav를 Gesticulator에 넣어서 얻은 .bvh
"""



from argparse import ArgumentParser
import io
import os
import subprocess
from argparse import ArgumentParser
from google.cloud import texttospeech

### STT
# from __future__ import division

import re
import sys

from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import speech
import pyaudio
# from six.moves import queue
from queue import Queue
import shutil


import torch
import librosa
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from gesticulator.model.model import GesticulatorModel
from gesticulator.interface.gesture_predictor import GesturePredictor
from gesticulator.visualization.motion_visualizer.generate_videos import visualize


'''DialoGPT'''
torch.set_grad_enabled(False)
# dialoPath = 'C:/Users/SOGANG/Documents/developer/I/gesticulator-master/demo/'

tokenizer = GPT2Tokenizer('nlp/medium/vocab.json', 'nlp/medium/merges.txt')

weights = torch.load('nlp/medium/medium_ft.pkl')
# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight", None)

cfg = GPT2Config.from_json_file('nlp/medium/config.json')
model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
model.load_state_dict(weights)
# if device_f == 'cuda':
#     model.half()
# model.to(device_f)
# model.eval()

weights = torch.load('nlp/medium/small_reverse.pkl')
# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight", None)

reverse_model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
reverse_model.load_state_dict(weights)
# if device_r == 'cuda':
#     reverse_model.half()
# reverse_model.to(device_r)
# reverse_model.eval()

device = torch.device('cpu')

end_token = torch.tensor([[50256]], dtype=torch.long)


def _get_response(output_token, past):
    out = torch.tensor([[]], dtype=torch.long, device= device)

    while True:
        output_token, past = model.forward(output_token, past=past)
        output_token = output_token[:, -1, :].float()
        indices_to_remove = output_token < torch.topk(output_token, 20)[0][..., -1, None]
        output_token[indices_to_remove] = -float('Inf')
        output_token = torch.multinomial(F.softmax(output_token, dim=-1), num_samples=1)

        out = torch.cat((out, output_token), dim=1)

        if output_token.item() == end_token.item():
            break

    return out, past


def _score_response(output_token, correct_token):
    inputs = torch.cat((output_token, correct_token), dim=1)
    mask = torch.full_like(output_token, -100, dtype=torch.long)
    labels = torch.cat((mask, correct_token), dim=1)

    loss, _, _ = reverse_model(inputs, labels=labels)

    return -loss.float()


def append_messages(old_list: list, new_list: list, truncate_length=64):
    for message in new_list:
        if message != '':
            input_token = tokenizer.encode(message, return_tensors='pt')
            input_token = torch.cat((input_token, end_token), dim=1)
            old_list.append(input_token)

    if len(old_list) == 0:
        old_list.append(end_token)

    # truncate
    total_length = 0
    for i, message in enumerate(reversed(old_list)):
        total_length += message.shape[1]
        if total_length > truncate_length:
            old_list[:] = old_list[-i:]


def generate_message(message_list: list, focus_last_message=True):
    total_input = torch.cat(message_list, dim=1).to(device)
    if focus_last_message:
        total_input_reversed = message_list[-1]
    else:
        total_input_reversed = torch.cat(list(reversed(message_list)), dim=1)

    past = None
    if total_input.shape[1] > 1:
        _, past = model(total_input[:, :-1])

    results = []
    for i in range(10):
        result = _get_response(total_input[:, -1:], past)
        score = _score_response(result[0].to(device), total_input_reversed.to(device))
        results.append(result + (score,))

    scores = torch.stack([x[2] for x in results], dim=0)
    winner = torch.multinomial(F.softmax(scores / 0.5, dim=0), num_samples=1).item()
    # winner = torch.argmax(scores, dim=0)

    out = results[winner][0]

    return tokenizer.decode(out.tolist()[0], skip_special_tokens=True)

# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('--audio', type=str, default="input/jeremy_howard.wav", help="path to the input speech recording")
#     # parser.add_argument('--text', type=str, default="input/jeremy_howard.json",
#     #                     help="one of the following: "
#     #                          "1) path to a time-annotated JSON transcription (this is what the model was trained with) "
#     #                          "2) path to a plaintext transcription, or " 
#     #                          "3) the text transcription itself (as a string)")
#     parser.add_argument('--video_out', '-video', type=str, default="output/generated_motion.mp4",
#                         help="the path where the generated video will be saved.")
#     parser.add_argument('--model_file', '-model', type=str, default="models/default.ckpt",
#                         help="path to a pretrained model checkpoint")
#     parser.add_argument('--mean_pose_file', '-mean_pose', type=str, default="../gesticulator/utils/mean_pose.npy",
#                         help="path to the mean pose in the dataset (saved as a .npy file)")
    
#     return parser.parse_args()



# print(type(my_response))
# append_messages(my_message_list, [my_response])




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
    motion = gp.predict_gestures(args.audio, my_response)

    # 3. Visualize the results
    motion_length_sec = int(motion.shape[1] / 20)

    visualize(motion.detach(), "agent.bvh", "temp.npy", "temp.mp4", 
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
    parser.add_argument('--audio', type=str, default="audio2.wav", help="path to the input speech recording")
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

'''TTS'''
def tts(text):
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text = text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code = 'en-US',
        ssml_gender = texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(
        audio_encoding = texttospeech.AudioEncoding.LINEAR16
    )
    response = client.synthesize_speech(
        input= synthesis_input, 
        voice = voice, 
        audio_config = audio_config)
    
    with open('audio2.wav', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "audio2.wav"')
        # src = "C:/Users/SOGANG/Desktop/inde_unity/gesticulator-master/demo/audio2.wav"
        # dst = "C:/Users/SOGANG/Desktop/inde_unity/Assets/Resources/Agentaudio/"
        
        # shutil.copy2(src, dst)
            


# Run all

if __name__ == "__main__":  
    my_message_list = []
    
    """ [DialoGPT] text -> response text """
    # avatarPath = 'C:/Users/SOGANG/Documents/developer/I/gesticulator-master/demo'
    f = open('text1.txt','r')
    input_text = f.read()
    print(f"input text = {input_text}")
    
    my_message = input_text
    append_messages(my_message_list, [my_message])
    try:
        print(f"my_message_list = {my_message_list}") 
        my_response = generate_message(my_message_list)
        print('bot >>', my_response)
        
        """ [TTS] response text -> response audio.wav """
        tts(my_response)
        # audio = 'audio2.wav'

        """ [Gesticulator] response text, response audio.wav -> gesture bvh """
        args = parse_args()
        main(args, my_response)
        os.rename("temp.bvh", "agent.bvh")
        # src = "C:/Users/SOGANG/Desktop/inde_unity/gesticulator-master/demo/agent.bvh"
        # dst = "C:/Users/SOGANG/Desktop/inde_unity/Assets/"
        # shutil.copy2(src, dst)
        
    except ValueError:
        my_response = generate_message(my_message_list)
        print('bot >>', my_response)
        print("ValueError")
        
        tts(my_response)

        args = parse_args()
        
        main(args, my_response)
        
    except IndexError:
        my_response = generate_message(my_message_list)
        print('bot >>', my_response)
        print("IndexError")
        
        tts(my_response)

        args = parse_args()
        
        main(args, my_response)
