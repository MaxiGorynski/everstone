import sys
import os

from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import os

import librosa
from flask import Flask, render_template, request, redirect, url_for, render_template_string, flash
from flask_login import LoginManager, current_user, login_required
from everstone.models import Recording, Transcript
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate, migrate
from markupsafe import Markup
import speech_recognition as sr
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import soundfile as sf  # Ensure soundfile is imported to write audio files
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

#Table of Contents #
### NLTK Data ad App Basics ###
### Uploads Folder and Database Basics ###
### E-X - Initialising the App/App Factory ###
### E1 - Route for Our Main Page ###
### E2 - Route to handle file uploading ###
### E3 - Route to display stored recordings ###
### E3A - Route to review recordings and transcript, incl. preprocessing, loading and generating webm files ###
### E4 - BM25 Search ###
### E4.2 - Embeddings ###
### E5 - User Login & Auth ###
### E0 - Marginal Utilities, including file deletion, file renaming ###

# Downloading NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize extensions from extensions.py
from everstone.extensions import db, bcrypt, login_manager, migrate

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if it doesn't exist

### E-X - Initialising the App/App Factory ###

# Define the app factory function. All routes contained within
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'EverstoneSki24'
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True) #Ensure the directory exists
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(basedir, 'instance', 'site.db')}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions with app context
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    login_manager.login_view = 'auth.login'

    # Import User model
    from everstone.models import User

    #Define user_loader function
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    from everstone.auth.routes import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    @app.errorhandler(404)
    def page_not_found(e):
        return "404 Error: The page you are trying to access does not exist."

    ################# Route for our main page (e1) #################
    @app.route('/')
    def index():
        return render_template('index.html')

    # Route to handle the file upload (e2)
    @app.route('/upload', methods=['POST'])
    @login_required
    def upload():
        try:
            # Sets the upload folder if not set
            app.config['UPLOAD_FOLDER'] = 'static/uploads'
            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)  # Ensure the directory exists

            #Does the audio component of the request exist?
            if 'audio' not in request.files:
                flash('No audio file part', 'danger')
                return redirect(url_for('index'))

            audio_file = request.files['audio']
            if audio_file.filename == '':
                flash('No selected file', 'danger')
                return redirect(url_for('index'))

            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)  # Creates the directory if it doesn't already exist

            # Construct the full file path and save the file
            if audio_file:
                filename = secure_filename(audio_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(filepath)

            # Create a new recording in the database, linked to current user
                new_recording = Recording(filename=audio_file.filename, user_id=current_user.id)
                db.session.add(new_recording)
                db.session.commit()

                flash('File uploaded successfully!', 'success')
                return redirect(url_for('recordings'))

            flash('Upload failed.', 'danger')
            return redirect(url_for('index'))
        except Exception as e:
            print(f"Error during upload: {e}")
            # Return the error message for debugging
            return f'Error: {str(e)}', 500

    #################### e3 and e3a zone ####################

    # Route to display stored recordings (e3)
    @app.route('/recordings')
    # @login_required
    def recordings():
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))

        # Query the database for recordings belonging to the current user
        user_recordings = Recording.query.filter_by(user_id=current_user.id).all()
        audio_files = [rec.filename for rec in user_recordings]

        return render_template('recordings.html', audio_files=audio_files)

    # Route to review recordings and transcript (e3a)
    @app.route('/review/<filename>', methods=['GET', 'POST'])
    @login_required
    def review_recording(filename):
        # Query the database to find
        recording = Recording.query.filter_by(filename=filename, user_id=current_user.id).first()
        if not recording:
            return "Recording not found."

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

            #Save the transcript to the database
            new_transcript = Transcript(content=transcript, recording_id=recording.id)
            db.session.add(new_transcript)
            db.session.commit()
            print("Transcript saved to database")

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
        # Load the audio file
        audio = AudioSegment.from_file(file_path)

        # Normalise the volume
        normalised_audio = audio.apply_gain(-audio.max_dBFS)

        # Export the normalised audio to a temporary file
        temp_file_path = 'temp.wav'
        normalised_audio.export(temp_file_path, format='wav')

        y, sr = librosa.load(temp_file_path, sr=None)

        y_denoised = nr.reduce_noise(y=y, sr=sr)

        denoised_file_path = 'denoised_temp.wav'
        sf.write(y_denoised, y_denoised, sr)

        return denoised_file_path

    # Path for storing transcripts data
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

        # Create a set for existing filenames
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
                # print(f"Saved transcripts_data.json with {len(transcripts_data)} entries.")
        else:
            print(f"Transcript for {filename} already exists.")

        #Saving transcript to database
        recording = Recording.query.filter_by(filename=filename).first()
        if recording:
            existing_transcript = Transcript.query.filter_by(recording_id=recording.id).first()
            if not existing_transcript:
                new_transcript = Transcript(
                    content=transcript_text,
                    recording_id=recording.id
                )
                db.session.add(new_transcript)
                db.session.commit()
                print(f"Transcript for recording {filename} saved to database.")
            else:
                print(f"Transcript for recording {filename} already exists.")
        else:
            print(f"Recording with filename {filename} not found in the database.")

    # Generates a transcript using SpeechRecognition
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

            # Save the transcript and metadata to JSON
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
            # print(transcripts_data)

        # Use a set to track unique filenames
        seen = set()
        unique_transcripts = []

        for entry in transcripts_data:
            if entry['filename'] not in seen:
                seen.add(entry['filename'])
                unique_transcripts.append(entry)

        return unique_transcripts

    def generate_transcripts_for_all_files():
        # List all .webm files in the UPLOAD_FOLDER
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.webm'):
                if not transcript_exists(filename):
                    print(f"Generating transcript for {filename}.")
                    generate_transcript_webm(filename)

    def transcript_exists(filename):
        # Load existing transcripts data
        transcripts_data = load_transcripts_data()

        # Check if the filename exists in the transcripts data
        return any(entry['filename'] == filename for entry in transcripts_data)

    # Load transcripts data once on app startup
    transcripts_data = load_transcripts_data()

    # Generate transcripts for all audio files upon startup
    generate_transcripts_for_all_files()

    #################### e4.1 zone, BM25 ####################

    # Route to pre-processing text function (e4)
    def preprocess_text(text):
        # Tokenisation
        tokens = nltk.word_tokenize(text.lower())
        # Removal of punctuation and stop words
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        return tokens

    # Preparation of transcripts data for BM25 search
    def prepare_bm25_data(transcripts_data):
        # Tokenisation of each transcript for BM25 search
        bm25_corpus = []
        for entry in transcripts_data:
            processed_text = preprocess_text(entry["transcript"])
            bm25_corpus.append(processed_text)
        return bm25_corpus

    # Preprocess and save BM25 data for later usage
    def load_and_prep_data():
        transcripts_data = load_transcripts_data()

        # Check if transcripts is empty
        if not transcripts_data:
            print("No transcripts found. Please add transcripts.")
            return [], None  # Return empty list and None for bm25

        bm25_corpus = prepare_bm25_data(transcripts_data)
        bm25 = BM25Okapi(bm25_corpus)  # Initialising BM25 with tokenised corpus
        # Print the structure of bm25_corpus to check it
        # print(bm25_corpus[:2])  # Print the first two entries to verify
        # Storing the preprocessed data
        return transcripts_data, bm25

    # Initialising preprocessing outside of the function, for global access, and so we only have to pre-process once,
    # ...instead of during every search
    transcripts_data, bm25 = load_and_prep_data()

    def test_bm25(query):
        preprocessed_query = preprocess_text(query)
        scores = bm25.get_scores(preprocessed_query)

        # Print scores to verify
        # for idx, score in enumerate(scores):
        # print(f"Transcript {idx + 1}: Score {score}")

    # test_bm25("test")

    def debug_bm25():
        # Minimal corpus example for direct BM25 testing
        mini_corpus = [["this", "is", "a", "test"], ["another", "test", "case"]]
        bm25_test = BM25Okapi(mini_corpus)
        sample_query = ["test"]

        # Get scores for sample query
        sample_scores = bm25_test.get_scores(sample_query)
        # print("Sample corpus scores:", sample_scores)

    debug_bm25()

    def bm25_search(query):
        query_tokens = preprocess_text(query)  # Process the query like we did transcripts
        bm25_scores = bm25.get_scores(query_tokens)  # Get relevance scores
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        # Retrieve the top results and corresponding transcript
        top_results = [(transcripts_data[i]['file_name'], transcripts_data[i]['transcript'], bm25_scores[i]) for i in
                       ranked_indices[:5]]
        return top_results

    def search_bm25(query, transcripts_data):
        # Preprocess the query
        preprocessed_query = preprocess_text(query)  # Tokenised as list
        scores = bm25.get_scores(preprocessed_query)  # Return a list of scores

        # Create sorted results
        results = sorted(
            zip(scores, transcripts_data),
            key=lambda x: x[0],
            reverse=True
        )

        return results

    #################### e4.2 zone, Embeddings ####################

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embeddings(transcripts_data):
        # Ensure transcripts_data is preprocessed or cleaned if needed
        for entry in transcripts_data:
            entry['embedding'] = embedding_model.encode(entry['transcript'])
        return transcripts_data

    def save_transcripts_with_embeddings(transcripts_data):
        # Convert ndarray to lists for JSON serialisation
        for entry in transcripts_data:
            if 'embedding' in entry:
                if isinstance(entry['embedding'], np.ndarray):
                    # Convert ndarray to list
                    entry['embedding'] = entry['embedding'].tolist()

        with open('transcripts_data.json', 'w') as file:
            json.dump(transcripts_data, file, indent=4)

    # Function to check if we have transcripts and generate them if not
    def ensure_transcripts_exist():
        # Load existing transcripts
        transcripts_data = load_transcripts_data()

        # If no transcripts are found, generate them
        if not transcripts_data:
            # Get all audio files from UPLOAD_FOLDER
            audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.webm')]

            for audio_file in audio_files:
                # Generate transcript for each audio file
                transcript = generate_transcript_webm(audio_file)
                print(f"Transcript for {audio_file}: {transcript}")

        return load_transcripts_data()

    # Load transcripts/guarantee they exist
    transcripts_data = ensure_transcripts_exist()

    # Generate embeddings and save transcripts with embeddings
    transcripts_data = generate_embeddings(transcripts_data)
    save_transcripts_with_embeddings(transcripts_data)

    # Creating a search endpoint within the app
    @app.route('/search', methods=['POST'])
    @login_required
    def search():
        query = request.form['query']
        transcripts_data = load_transcripts_data()

        # Filtering the transcripts data to only include entries owned by the user
        user_transcripts_data = [entry for entry in transcripts_data if
                                 'user_id' in entry and entry['user_id'] == current_user.id]
        results = search_bm25(query, user_transcripts_data)

        return render_template('search_results.html', results=results, search_term=query)

    #################### E5 - User Login ####################

    def load_user(user_id):
        return User.query.get(int(user_id))

    #################### E0 - Marginal utilities ####################

    # Route to handle file renaming
    @app.route('/rename_file', methods=['POST'])
    @login_required
    def rename_file():
        old_name = request.form['old_name']
        new_name = request.form['new_name']

        if not new_name.endswith('.webm'):
            new_name += '.webm'

        upload_folder = app.config['UPLOAD_FOLDER']
        old_file_path = os.path.join(upload_folder, old_name)
        new_file_path = os.path.join(upload_folder, new_name)

        # Check if the old file exists before renaming
        if not os.path.exists(old_file_path):
            return f"File '{old_name}' does not exist.", 404

        if os.path.exists(new_file_path):
            return f"File '{new_name}' already exists. Choose a different name.", 400

        os.rename(old_file_path, new_file_path)

        from everstone.models import Recording
        recording = Recording.query.filter_by(filename=old_name, user_id=current_user.id).first()

        if recording:
            recording.filename = new_name
            db.session.commit()
            print(f"Recording renamed in the database: {old_name} -> {new_name}")
        else:
            return f"Recording with filename '{old_name}' not found in the database.", 404

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

    # Route to handle to file deletions
    @app.route('/delete_recording/<filename>', methods=['POST'])
    @login_required
    def delete_recording(filename):
        try:
            #Delete audio file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted audio file {file_path}.")
            else:
                print(f"Audio file {file_path} not found.")

            #Remove recording and transcript entries from the database
            from everstone.models import Recording, Transcript
            recording = Recording.query.filter_by(filename=filename, user_id=current_user.id).first()

            if recording:
                #Delete associated transcript, if it exists
                transcript = Transcript.query.filter_by(recording_id=recording.id).first()
                if transcript:
                    db.session.delete(transcript)
                    print(f"Delete transcript for recording {filename}")

                #Delete recording proper
                db.session.delete(recording)
                db.session.commit()
                print(f"Deleted recording {filename} from database.")
            else:
                print(f"Recording with filename '{filename}' not found in database.")

            flash(f"Recording '{filename}' and its transcript have been deleted.", 'success')
            return redirect(url_for('recordings'))
        except Exception as e:
            print(f"Error during deletion: {str(e)}")
            flash(f"A error occured while deleting recording '{filename}'.", 'danger')
            return redirect(url_for('recordings'))
    return app

# Create the Flask app using the factory pattern
app = create_app()

# Print registered routes immediately after creation
print("\nRegistered Routes Immediately After App Creation:")
for rule in app.url_map.iter_rules():
    print(f"Endpoint: {rule.endpoint}, URL: {rule}\n")


if __name__ == '__main__':
    app = create_app()  # Calling our factory function
    app.run(host='0.0.0.0', port=5001, debug=True)  # Start the app

