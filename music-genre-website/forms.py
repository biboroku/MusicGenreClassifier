from flask_wtf import FlaskForm
#from flask_login import current_user
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, IntegerField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo, Length, Regexp

class TrainingForm(FlaskForm):
    playlist1 = StringField('Spotify playlist 1', validators=[DataRequired()])
    playlist2 = StringField('Spotify playlist 2', validators=[DataRequired()])
    classifier = SelectField('Choose classifier type', choices=[('K Nearest Neighbor', 'knn'), 
                                                                ('Decision Tree', 'tree'), 
                                                                ('Support Vector Machine', 'svm')])
    submit = SubmitField('Train my classifier now!')

class PredictForm(FlaskForm):
    target_playlist = StringField('Predict the genre of this playlist', validators=[DataRequired()])
    submit = SubmitField('Tell me now!')
