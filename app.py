import os

from flask import Flask, render_template, request, redirect, url_for, render_template_string

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


# Route to display stored recordings (e3)
@app.route('/recordings')
def recordings():
    # List all files in the uploads directory
    audio_files = os.listdir(UPLOAD_FOLDER)
    # Filter to only include audio files (to be expanded later)
    audio_files = [f for f in audio_files if f.endswith('.webm')]

    return render_template('recordings.html', audio_files=audio_files)

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
