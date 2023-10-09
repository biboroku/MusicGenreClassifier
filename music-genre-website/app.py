from flask import Flask, render_template, request
from forms import TrainingForm
from flask import session
from flask_bootstrap import Bootstrap4
from classifier import getSpotifyPlaylist, downloadPlaylistAsExcel, trainKnnClassifier, trainDecisionTreeClassifier
import secrets

app = Flask(__name__)
bootstrap = Bootstrap4(app)

# Souta is cool

# Spotify credentials
client_id = "08afbd9799f847a68d2e28997f98f0aa"
client_secret = "2cd6d23b1eed44e6ab37ef7abd9eb809"

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = secrets.token_hex(16)

knn_model = None
tree_model = None


@app.route("/", methods = ['GET', 'POST'])
def index():
    form = TrainingForm()
    if form.validate_on_submit():
        playlist1 = form.playlist1.data
        playlist2 = form.playlist2.data
        classifier = form.classifier.data
        print(playlist1)
        print(playlist2)
        return render_template('index.html', form=form)
    return render_template('index.html', form=form)

@app.route("/", methods = ['GET'])
def prediction():
    return None
