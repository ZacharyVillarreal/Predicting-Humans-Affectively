import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_daq as daq
import base64
import os
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pickle
import numpy as np
import subprocess

import warnings
warnings.filterwarnings('ignore')


UPLOAD_DIRECTORY = "assets/"
import base64
import os
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])


# @server.route("/download/<path:path>")
# def download(path):
#     """Serve a file from the upload directory."""
#     return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = html.Div(
    [
        html.H1("Emotion Detection App", style={'textAlign':'center'}),
        dbc.Row([
            dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "40%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,),
        ], align = 'right'),
        html.H2(
               id="emotion",
               children = "Emotion Detected: Waiting for an input...",
               style = {'display': 'inline-block'}
               ),
        html.H3(
                id="sex",
                children = "Sex Detected: Waiting for an input..." 
               ),
#         html.Ul(id="file-list"),
        html.Label(id='l1', children=''),
        html.Audio(
                id='a1', 
                controls = True, 
                autoPlay = False,
                style = {'display':'inline-block'}
                ),
        html.Img(
                id='img1',
                style={'height':'550px'}
                )

        
    ],
)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)




@app.callback(
    [Output("a1", "src"), Output("l1", "children"), Output("img1", "src"), 
     Output("emotion", "children"), Output("sex", "children")],
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""
    print("Update output called.")
    url, image, emotion, sex = get_file(uploaded_filenames, uploaded_file_contents)
    print("Url is:", url)
    return url, url, image, "Emotion Detected: " + emotion.upper(), "Sex Detected: " + sex.upper()

def get_file(uploaded_filenames, uploaded_file_contents):
    url = '/assets/male_angry.wav'
    image = '/assets/nathan.jpg'
    emotion = "Waiting for an input..."
    sex = "Waiting for an input..."
    
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            if "wav" in name or "m4a" in name or "mp3" in name:
                save_file(name, data)
                print('We recieved an audio file.')
                image = app.get_asset_url(analyze_sound(name)[0])
                emotion, sex = analyze_sound(name)[1:]
                url = "/assets/" + name
            else:
                print("We received an image file.")
                save_file(name, data)
                url = app.get_asset_url(analyze_file(name)[0])
                emotion, sex = analyze_file(name)[1:]
                image = "/assets/" + name
    return url, image, emotion, sex
                
def analyze_file(name):
    cmd = ['python', 'image_analyzer.py', name]
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    outputs = []
    for line in p.stdout.readlines():
        print('line:', line)
        outputs.append(line)
#     print('Call complete: ', outputs)
    audio_file = outputs[-1].strip().decode()
    sex = outputs[-2].strip().decode()
    emotion = outputs[-3].strip().decode()
#     print("File test: ",audio_file)
    return audio_file, emotion, sex

def analyze_sound(name):
    cmd = ['python', 'audio_analyzer.py', name]
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    outputs = []
    for line in p.stdout.readlines():
        print('line:', line)
        outputs.append(line)
#     print('Call complete: ', outputs)
    image_file = outputs[-1].strip().decode()
    sex = outputs[-3].strip().decode()
    emotion = outputs[-2].strip().decode()
#     print("File test: ",image_file)
    return image_file, emotion, sex
    
    
    
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8899)