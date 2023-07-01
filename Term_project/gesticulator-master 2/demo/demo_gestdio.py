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


import torch
import librosa
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from gesticulator.model.model import GesticulatorModel
from gesticulator.interface.gesture_predictor import GesturePredictor
from gesticulator.visualization.motion_visualizer.generate_videos import visualize


'''DialoGPT'''
torch.set_grad_enabled(False)

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
    print(f"my_response = {my_response}")
    motion = gp.predict_gestures(args.audio, my_response)

    # 3. Visualize the results
    motion_length_sec = int(motion.shape[1] / 20)

    visualize(motion.detach(), "temp.bvh", "temp.npy", "temp.mp4", 
              start_t = 0, end_t = motion_length_sec, 
              data_pipe_dir = 'C:/Users/SOGANG/Documents/developer/inde/gesticulator-master/gesticulator/utils/data_pipe.sav')

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
    parser = ArgumentParser()
    parser.add_argument('--audio', type=str, default="output.wav", help="path to the input speech recording")
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
    
    with open('output.wav', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content weitten to file "output.wav"')
            

'''STT code - mic input version
# Audio recording parameters
RATE = 44100
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except Queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            text = transcript + overwrite_chars
            print(text)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0
            return text


def stt():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'en-US'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        input_text = listen_print_loop(responses)
        return input_text
'''

'''STT - .wav version'''
def stt():
    # Instantiates a client
    # [START migration_client]
    client = speech.SpeechClient()
    # [END migration_client]

    # The name of the audio file to transcribe
    file_name = os.path.join(
        os.path.dirname(__file__),
        '.',
        'output.wav')

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
        print(f"output.wav = {text}")
    # [END speech_quickstart]
    return text



# Run all

if __name__ == "__main__":  
    my_message_list = []
    
    """ [STT] audio.wav -> text """
    input_text = stt()[0]
    print(f"input text = {input_text}")
if True:
    # my_message = input("Input your text : " )
    my_message = input_text
    append_messages(my_message_list, [my_message])
    try:
        """ [DialoGPT] text -> response text """
        print(f"my_message_list = {my_message_list}") 
        my_response = generate_message(my_message_list)
        # print(f"my_response = {my_response}")
        print('bot >>', my_response)
        
        """ [TTS] response text -> response audio.wav """
        tts(my_response)
        # audio = 'output.wav'

        """ [Gesticulator] response text, response audio.wav -> gesture bvh """
        print(type(my_response), f"my_response = {my_response}")
        args = parse_args()
        main(args,  my_response)
        
    except ValueError:
        my_response = generate_message(my_message_list)
        print('bot >>', my_response)
        print("ValueError")
        
        tts(my_response)
        # audio = 'output.wav'
        args = parse_args()
        
        main(args,  my_response)
        
    except IndexError:
        my_response = generate_message(my_message_list)
        print('bot >>', my_response)
        print("IndexError")
        
        tts(my_response)
        # audio = 'output.wav'
        args = parse_args()
        
        main(args,  my_response)
