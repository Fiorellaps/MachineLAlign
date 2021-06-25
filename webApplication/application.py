# -*- coding: utf-8 -*-
"""
@author fiorellaps
@since 13/04/21
Program for creating the app

"""

from flask import Flask, flash, url_for, request, session, redirect
from flask import render_template
import align_sequences


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alinear', methods=['POST'])
def alinear_secuencias():
    secuencia = request.form.get("secuencia")
    content_decisiontree = ''
    content_adaboost = ''
    content_knn = ''
    content_gradientboosting = ''
    content_randomforest = ''
    content = dict()
    if('adaboost' in request.form):
        output = align_sequences.alignSequences(secuencia,'adaboost')
        output_name = output[0]
        tool_name = output [1]
        content['AdaBoost'] = {'content' : output_name, 'out_file': output_name, 'tool': tool_name}
    
    if('decisiontree' in request.form):
        output = align_sequences.alignSequences(secuencia,'decisiontree')
        output_name = output[0]
        tool_name = output [1]
        content['Decision Tree'] = {'content': output_name, 'out_file': output_name, 'tool': tool_name}
    if('knn' in request.form):
        output = align_sequences.alignSequences(secuencia,'knn')
        output_name = output[0]
        tool_name = output [1]
        content['KNN'] = {'content' : output_name , 'out_file': output_name, 'tool': tool_name}

    if('gradientboosting' in request.form):
        output = align_sequences.alignSequences(secuencia,'gradientboosting')
        output_name = output[0]
        tool_name = output [1]
        content['Gradient Boosting'] = {'content' : output_name , 'out_file': output_name, 'tool': tool_name}

    if('randomforest' in request.form):
        output = align_sequences.alignSequences(secuencia,'randomforest')
        output_name = output[0]
        tool_name = output [1]
        content['Random Forest'] = {'content':output_name, 'out_file': output_name, 'tool': tool_name}
    
    if('cnnbilstm' in request.form):
        output = align_sequences.alignSequences(secuencia,'cnnbilstm')
        output_name = output[0]
        tool_name = output [1]
        content['CNN BiLSTM'] = {'content':output_name, 'out_file': output_name, 'tool': tool_name}
    
    if('cnn' in request.form):
        output = align_sequences.alignSequences(secuencia,'cnn')
        output_name = output[0]
        tool_name = output [1]
        content['CNN'] = {'content':output_name, 'out_file': output_name, 'tool': tool_name}
    
    return render_template('alignment.html', content = content)
    
if __name__ == '__main__':
    app.run(debug=True)

