import json
import os
from flask import Flask, render_template, request, redirect, url_for, render_template_string
import speech_recognition as sr
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import soundfile as sf  # Ensure soundfile is imported to write audio files
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from rank_bm25 import BM25Okapi

#Dowloading NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Define the uploads folder where recordings will be stored
UPLOAD_FOLDER = 'static/uploads'  # Use static/uploads for consistent access
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if it doesn't exist


# Route for our main page (e1)
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle the file upload (e2)
@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'audio' not in request.files:
            return 'No file part', 400

        file = request.files['audio']

        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)  # Creates the directory if it doesn't already exist

        # Construct the full file path
        file_path = os.path.join(upload_folder, file.filename)

        # Save the file
        file.save(file_path)

        return 'File uploaded successfully', 200
    except Exception as e:
        # Return the error message for debugging
        return f'Error: {str(e)}', 500

#################### e3 and e3a zone ####################

# Route to display stored recordings (e3)
@app.route('/recordings')
def recordings():
    # List all files in the uploads directory
    audio_files = os.listdir(UPLOAD_FOLDER)
    # Filter to only include audio files (to be expanded later)
    audio_files = [f for f in audio_files if f.endswith('.webm')]

    return render_template('recordings.html', audio_files=audio_files)

#Route to review recordings and transcript (e3a)
@app.route('/review/<filename>', methods=['GET', 'POST'])
def review_recording(filename):
    transcript = None
    if request.method == 'POST':
        # Generate transcript only if the button is clicked
        transcript = generate_transcript_webm(filename)

    return render_template('review_recording.html', filename=filename, transcript=transcript)


def preprocess_audio(file_path):
    #Load the audio file
    audio = AudioSegment.from_file(file_path)

    #Normalise the volume
    normalised_audio = audio.apply_gain(-audio.max_dBFS)

    #Export the ormalised audio to a temporary file
    temp_file_path = 'temp.wav'
    normalised_audio.export(temp_file_path, format='wav')

    y, sr = librosa.load(temp_file_path, sr=None)

    y_denoised = nr.reduce_noise(y=y, sr=sr)

    denoised_file_path = 'denoised_temp.wav'
    sf.write(y_denoised, y_denoised, sr)

    return denoised_file_path

#Path for storing transcripts data
TRANSCRIPTS_DATA_FILE = 'transcripts_data.json'

def save_transcript_data(filename, transcript_text):
    url = f"/review/{filename}"
    json_file_path = 'transcripts_data.json'

    # Load existing data or initialize an empty list
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as file:
                transcripts_data = json.load(file)
        except json.JSONDecodeError:
            # If file is empty or corrupted, initialize as an empty list
            transcripts_data = []
    else:
        transcripts_data = []

    # Add the new transcript entry
    new_entry = {
        "filename": filename,
        "url": url,
        "transcript": transcript_text
    }
    transcripts_data.append(new_entry)

    # Write back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(transcripts_data, file, indent=4)

#Generates a transcript using SpeechRecognition
def generate_transcript_webm(filename):
    recogniser = sr.Recognizer()
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        audio = AudioSegment.from_file(file_path, format="webm")
        wav_file_path = os.path.join(UPLOAD_FOLDER, 'temp.wav')
        audio.export(wav_file_path, format="wav")

        with sr.AudioFile(wav_file_path) as source:
            audio_data = recogniser.record(source)

        transcript = recogniser.recognize_google(audio_data)

        # Clean up the temporary file
        os.remove(wav_file_path)

        #Save the transcript and metadata to JSON
        save_transcript_data(filename, transcript)

        return transcript
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error processing the file: {e}"

def load_transcripts_data(filepath="transcripts_data.json"):
    with open(filepath, 'r') as file:
        transcripts_data = json.load(file)
    return transcripts_data

#Load data once on app startup
transcripts_data = load_transcripts_data()

#################### e4 zone ####################

#Route to pre-processing text function (e4)
def preprocess_text(text):
    #Tokenisation
    tokens = nltk.word_tokenize(text.lower())
    #Removal of punctuation and stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

#Preparation of transcripts data for BM25 search
def prepare_bm25_data(transcripts_data):
    #Tokenisation of each transcript for BM25 search
    bm25_corpus = []
    for entry in transcripts_data:
        processed_text = preprocess_text(entry["transcript"])
        bm25_corpus.append("".join(processed_text))
    return bm25_corpus

#Preprocess and save BM25 data for later usage
def load_and_prep_data():
    transcripts_data = load_transcripts_data()
    bm25_corpus = prepare_bm25_data(transcripts_data)

    #Storing the preprocessed data
    return transcripts_data, bm25_corpus

#Initialising preprocessing outside of the function, for global access, and so we only have to pre-process once,
#...instead of during every search
transcripts_data, bm25_corpus = load_and_prep_data()

#Initialising bm25 with preprocessed corpus
bm25 = BM25Okapi(bm25_corpus)

def bm25_search(query):
    query_tokens = preprocess_text(query) #Process the query like we did transcripts
    bm25_scores = bm25.get_scores(query_tokens) #Get relevance scores
    ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse = True)

    #Retrieve the top results and corresponding transcript
    top_results = [(transcripts_data[i]['file_name'],transcripts_data[i]['transcript'], bm25_scores[i]) for i in ranked_indices[:5]]
    return top_results

def search_bm25(query, transcripts_data):
    #Preprocess the query
    preprocessed_query = preprocess_text(query)
    query_tokens = preprocessed_query.split()

    #Prepare the corpus for BM25
    corpus = [entry['trascript'] for entry in transcripts_data]
    bm25 = BM25Okapi([preprocess_text(test).split() for text in corpus])

    #Get scores for the query
    scores = bm25.get_scores(query_tokens)

    #Combine scores with the original data
    results = sorted(zip(scores, transcripts_data), reverse=True, key=lambda x: x[0])
    return results

#Create a search endpoint within the app

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_bm25(query, transcripts_data)

    return render_template('search_results.html', results=results)

#################### Marginal utilities ####################

#Route to handle file renaming
@app.route('/rename_file', methods=['POST'])
def rename_file():
    old_name = request.form['old_name']
    new_name = request.form['new_name']

    if not new_name.endswith('.webm'):
        new_name += '.webm'

    upload_folder = 'static/uploads'
    old_file_path = os.path.join(upload_folder, old_name)
    new_file_path = os.path.join(upload_folder, new_name)

    # Check if the old file exists before renaming
    if os.path.exists(old_file_path):
        # Check if the new file already exists
        if os.path.exists(new_file_path):
            return f"File '{new_name}' already exists. Choose a different name.", 400
        os.rename(old_file_path, new_file_path)  # Rename the file
        return render_template_string('''
            <html>
                <head>
                    <meta http-equiv="refresh" content="2;url=recordings">
                </head>
                <body>
                    <h1>File renamed successfully!</h1>
                    <p> You will be redirect in 3,2...</p>
                </body>
            </html>
        ''')
    else:
        return f"File '{old_name}' does not exist.", 404

if __name__ == '__main__':
    app.run(debug=True)
