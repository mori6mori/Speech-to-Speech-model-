import torch
import numpy as np
import soundfile as sf
from transformers import pipeline
import nbimporter
from STS_model import speech_to_speech_translation
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import matplotlib
from datasets import load_dataset
matplotlib.use('Agg')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)

def translate(audio):
    #translate to any english (default)
    outputs = pipe(audio, max_new_tokens=256, generate_kwargs={"task": "translate"})
    #transcribe to any language
    #outputs = pipe(audio, max_new_tokens=256, generate_kwargs={"task": "transcribe", "language": "es"})
    return outputs["text"]
#Text to Speech 


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = embeddings_dataset[7306]["xvector"]
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        #-> tokenized text (input ids) -> SpeechT5 model 
        #placing each on the accelerator device if available
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu()



target_dtype = np.int16
max_range = np.iinfo(target_dtype).max

def load_audio_file(file_path):
    # Load and return audio data as a numpy array
    audio_data, sampling_rate = sf.read(file_path)
    return audio_data, sampling_rate

def speech_to_speech_translation(audio_file_path):
    # Load the audio file
    audio_data, sampling_rate = load_audio_file(audio_file_path)
    
    # Now audio_data can be passed to the translate function
    translated_text = translate(audio_data)
    print(translated_text)  # Show translated text (if translation is what you're doing after transcription)
    translated_audio = synthesise(translated_text)
    translated_audio = (translated_audio.numpy() * max_range).astype(np.int16)
    sample_rate = 16000  
    return translated_audio, sample_rate


import gradio as gr
demo = gr.Blocks()

mic_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
)

file_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(sources=["upload"], type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
)

with demo:
    gr.TabbedInterface([mic_translate, file_translate], ["Microphone", "Audio File"])

demo.launch(debug=True)