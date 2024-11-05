import json
import os

import librosa
from flask import Flask, render_template, request, redirect, url_for, render_template_string, flash
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from auth.routes import auth_bp
from markupsafe import Markup
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
from sentence_transformers import SentenceTransformer

#Dowloading NLTK data
nltk.download('punkt')
nltk.download('stopwords')

######## App Basics ########
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
login_manager = LoginManager()
login_manager.init_app(app)

# Define the uploads folder where recordings will be stored
UPLOAD_FOLDER = 'static/uploads'  # Use static/uploads for consistent access
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if it doesn't exist

######## Database Basics ########

#Configuring the DB in SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///everstone.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#Initialising the DB
db = SQLAlchemy(app)

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
    # Load existing transcripts data
    transcripts_data = load_transcripts_data()
    # Log to verify function is being entered
    print("Entering review_recording with filename: ", filename)

    # Get the search term from the query parameter
    search_term = request.args.get('search_term', '')
    print("Search term received: ", search_term)

    # Check if the transcript already exists for the specified file
    transcript = next((entry['transcript'] for entry in transcripts_data if entry['filename'] == filename), None)
    print("Initial transcript fetched:", transcript)

    # If no transcript is found, generate it
    if not transcript:
        transcript = generate_transcript_webm(filename)
        # Add the new transcript data to the JSON file
        save_transcript_data(filename, transcript)
        print("Transcript generated:", transcript)

    # Highlight the search term in transcript if present
    if search_term:
        highlighted_transcript = transcript.replace(search_term, f"<b>{search_term}</b>")
        highlighted_transcript = Markup(highlighted_transcript)  # Mark the text as safe HTML for rendering
        print("Highlighted transcript:", highlighted_transcript)
    else:
        highlighted_transcript = transcript
        print("No search term to highlight. Transcript unchanged.")

    # Return the highlighted transcript instead of the original transcript
    return render_template('review_recording.html', filename=filename, transcript=highlighted_transcript)




def preprocess_audio(file_path):
    #Load the audio file
    audio = AudioSegment.from_file(file_path)

    #Normalise the volume
    normalised_audio = audio.apply_gain(-audio.max_dBFS)

    #Export the normalised audio to a temporary file
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

    #Create a set for existing filenames
    existing_filenames = {entry['filename'] for entry in transcripts_data}

    if filename not in existing_filenames:
        new_entry = {
        "filename": filename,
        "url": url,
        "transcript": transcript_text,
        "embedding": []
    }
        transcripts_data.append(new_entry)
        print(f"Adding new entry, {new_entry}")

        # Write back to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(transcripts_data, file, indent=4)
            #print(f"Saved transcripts_data.json with {len(transcripts_data)} entries.")
    else:
        print(f"Transcript for {filename} already exists.")

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
        print(f"Generated transcript: {transcript}")

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
        #print(transcripts_data)

    #Use a set to track unique filenames
    seen = set()
    unique_transcripts = []

    for entry in transcripts_data:
        if entry['filename'] not in seen:
            seen.add(entry['filename'])
            unique_transcripts.append(entry)

    return unique_transcripts

def generate_transcripts_for_all_files():
    #List all .webm files in the UPLOAD_FOLDER
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.webm'):
            if not transcript_exists(filename):
                print(f"Generating transcript for {filename}.")
                generate_transcript_webm(filename)

def transcript_exists(filename):
    #Load existing transcripts data
    transcripts_data = load_transcripts_data()

    #Check if the filename exists in the transcripts data
    return any(entry['filename'] == filename for entry in transcripts_data)

#Load transcripts data once on app startup
transcripts_data = load_transcripts_data()

#Generate transcripts for all audio files upon startup
generate_transcripts_for_all_files()

#################### e4.1 zone, BM25 ####################

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
        bm25_corpus.append(processed_text)
    return bm25_corpus

#Preprocess and save BM25 data for later usage
def load_and_prep_data():
    transcripts_data = load_transcripts_data()

    #Check if transcripts is empty
    if not transcripts_data:
        print("No transcripts found. Please add transcripts.")
        return [], None #Return empty list and None for bm25

    bm25_corpus = prepare_bm25_data(transcripts_data)
    bm25 = BM25Okapi(bm25_corpus) #Initialising BM25 with tokenised corpus
    # Print the structure of bm25_corpus to check it
    #print(bm25_corpus[:2])  # Print the first two entries to verify
    #Storing the preprocessed data
    return transcripts_data, bm25



#Initialising preprocessing outside of the function, for global access, and so we only have to pre-process once,
#...instead of during every search
transcripts_data, bm25 = load_and_prep_data()

def test_bm25(query):
    preprocessed_query = preprocess_text(query)
    scores = bm25.get_scores(preprocessed_query)

    #Print scores to verify
    #for idx, score in enumerate(scores):
        #print(f"Transcript {idx + 1}: Score {score}")

#test_bm25("test")

def debug_bm25():
    #Minimal corpus example for direct BM25 testing
    mini_corpus = [["this", "is", "a", "test"], ["another", "test", "case"]]
    bm25_test = BM25Okapi(mini_corpus)
    sample_query = ["test"]

    #Get scores for sample query
    sample_scores = bm25_test.get_scores(sample_query)
    #print("Sample corpus scores:", sample_scores)

debug_bm25()

def bm25_search(query):
    query_tokens = preprocess_text(query) #Process the query like we did transcripts
    bm25_scores = bm25.get_scores(query_tokens) #Get relevance scores
    ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse = True)

    #Retrieve the top results and corresponding transcript
    top_results = [(transcripts_data[i]['file_name'],transcripts_data[i]['transcript'], bm25_scores[i]) for i in ranked_indices[:5]]
    return top_results

def search_bm25(query, transcripts_data):
    #Preprocess the query
    preprocessed_query = preprocess_text(query) #Tokenised as list
    scores = bm25.get_scores(preprocessed_query) #Return a list of scores

    #Create sorted results
    results = sorted(
        zip(scores, transcripts_data),
        key= lambda x: x[0],
        reverse=True
    )

    return results

#################### e4.2 zone, Embeddings ####################

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(transcripts_data):
    #Ensure transcripts_data is preprocessed or cleaned if needed
    for entry in transcripts_data:
        entry['embedding'] = embedding_model.encode(entry['transcript'])
    return transcripts_data

def save_transcripts_with_embeddings(transcripts_data):
    #Convert ndarray to lists for JSON serialisation
    for entry in transcripts_data:
        if 'embedding' in entry:
            if isinstance(entry['embedding'], np.ndarray):
                #Convert ndarray to list
                entry['embedding'] = entry['embedding'].tolist()

    with open('transcripts_data.json', 'w') as file:
        json.dump(transcripts_data, file, indent=4)

#Function to check if we have transcripts and generate them if not
def ensure_transcripts_exist():
    #Load existing transcripts
    transcripts_data = load_transcripts_data()

    #If no transcripts are found, generate them
    if not transcripts_data:
        #Get all audio files from UPLOAD_FOLDER
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.webm')]

        for audio_file in audio_files:
            #Generate transcript for each audio file
            transcript = generate_transcript_webm(audio_file)
            print(f"Transcript for {audio_file}: {transcript}")

    return load_transcripts_data()

#Load transcripts/guarantee they exist
transcripts_data = ensure_transcripts_exist()

#Generate embeddings and save transcripts with embeddings
transcripts_data = generate_embeddings(transcripts_data)
save_transcripts_with_embeddings(transcripts_data)

#Creating a search endpoint within the app
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_bm25(query, transcripts_data)

    return render_template('search_results.html', results=results, search_term=query)

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

#Route to handle to file deletions
@app.route('/delete_recording/<filename>', methods=['POST'])
def delete_recording(filename):
    #Delete the audio file
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted audio file {file_path}")

    #Load the existing transcript data

    with open (TRANSCRIPTS_DATA_FILE, 'r') as file:
        transcripts_data = json.load(file)

    #Filter out the entry for the deleted file
    transcripts_data = [entry for entry in transcripts_data if entry['filename'] != filename]

    #Write the updated transcripts data back to the transcripts data file
    with open (TRANSCRIPTS_DATA_FILE, 'w') as file:
        json.dump(transcripts_data, file, indent=4)
    print(f"Removed transcript for {filename}")

    flash(f"Recording {filename} and its transcript have been deleted.")
    return redirect(url_for('recordings'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

