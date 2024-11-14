from PyPDF2 import PdfReader
from flask import render_template, url_for, flash, redirect, request,session,send_file
from questionpapergenerator import app, users_collection , pdf_collection
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import torch,random
import nltk
from nltk.corpus import wordnet as wn
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import datetime
import os
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from io import BytesIO
import io
from bson import ObjectId
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor
import nltk
import ssl
from rake_nltk import Rake

#Extraction from PDF
def extract_text_from_pdf(file_path):
   reader = PdfReader(file_path)
   text = ''
   for page in reader.pages:
       text += page.extract_text()
   return text


#Preprocessing of text
def preprocess_text(text, segment_length=1700):
   # Remove leading and trailing whitespace
   text = text.strip()


   # Replace bullet points with a space
   text = re.sub(r'\s*•\s*', ' ', text)


   # Replace newlines and multiple whitespaces with a single space
   text = ' '.join(text.split())


   # Split the text into segments of specified length
   segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]


   return segments


#Code for summarization
from transformers import T5ForConditionalGeneration,T5TokenizerFast
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5TokenizerFast.from_pretrained('t5-base')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)


def summarizer(text_segments, model, tokenizer):
   summaries = []


   for text_segment in text_segments:
       text = text_segment.strip().replace("\n", " ")
       text = "summarize: " + text
       max_len = 512
       encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)


       input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]


       outs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             early_stopping=True,
                             num_beams=3,
                             num_return_sequences=1,
                             no_repeat_ngram_size=2,
                             min_length=75,
                             max_length=300)


       dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
       summary = dec[0]
       summary = summary.strip()
       summaries.append(summary)


   return summaries


#Keyword Extraction


def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN','NOUN'}
        #pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)


        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out

def get_keywords(summary_texts):
    keyword_processor = KeywordProcessor()

    keywords_found = []
    for summarized_text in summary_texts:
        keywords_found = keyword_processor.extract_keywords(summarized_text)
            
    return keywords_found




def extract_keywords_rake(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()






#Question Generation
checkpoint = "t5-base"
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)


model = AutoModelForSeq2SeqLM.from_pretrained("rohan-jp1/t5-end2end-questions-generation")


import random


def hf_run_model(input_list, num_return_sequences=8, num_questions=2, max_sequence_length=512, generator_args=None):
   if generator_args is None:
       generator_args = {
           "max_length": max_sequence_length,
           "num_beams": 10,
           "length_penalty": 1.5,
           "no_repeat_ngram_size": 6,
           "early_stopping": True,
           "temperature": 0.8,  # Adjust the temperature value (higher values for more randomness)
           "top_k": 50,  # Adjust the top_k value (higher values for more diverse output)
           "top_p": 0.95  # Adjust the top_p value (lower values for more focused output)
       }
  
   generated_questions = []
   unique_questions = set()


   #creating tensors of each input
   for input_string in input_list:
       input_string = "generate questions: " + input_string + " </s>"
       input_ids = tokenizer.encode(input_string, truncation=True, max_length=max_sequence_length, return_tensors="pt")


       # Generate questions using the model
       res = model.generate(input_ids, **generator_args, num_return_sequences=num_return_sequences)
       output = tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=True)


       segment_questions = []
       for sequence in output:
           sequence = sequence.split("<sep>")
           questions = [question.strip() + "?" for question in sequence[0].split("?") if question.strip()]
           segment_questions.extend(questions[:num_questions])  # Selecting the desired number of questions from each segment


       # Filter out single-word questions for each segment
       segment_questions = [question for question in segment_questions if len(question.split()) > 1]
       generated_questions.extend(segment_questions)


   # Randomly sample questions until reaching the desired number of non-repeated questions
   while len(unique_questions) < num_questions * len(input_list):  # Generating questions from each segment
       question = random.choice(generated_questions)
       generated_questions.remove(question)
       if question not in unique_questions:
           unique_questions.add(question)


   return list(unique_questions)


#Making PDF


def convert_list_to_pdf_with_template(data_list, output_file,subject_name):
   # Create the PDF canvas
   c = canvas.Canvas(output_file, pagesize=letter)


   # Set the font and size
   c.setFont("Helvetica", 12)


   # Add the template or background image
   template_path = 'template.png'
   c.drawImage(template_path, 0, 0, width=letter[0], height=letter[1])


   # Set up paragraph styles
   styles = getSampleStyleSheet()
   paragraph_style = ParagraphStyle(
       'normal',
       parent=styles['Normal'],
       textColor=colors.black,
       fontSize=12,
       leading=16  # Adjust the leading for more spacing between lines
   )


   # Write the list elements to the PDF
   y = 550  # Starting y position
   index = 1
   spacing = 20  # Fixed spacing between paragraphs


   for item in data_list:
       text = f"{index}) {item}"
       p = Paragraph(text, style=paragraph_style)
       p.wrapOn(c, 400, 0)


       # Check if there's enough space on the page for the paragraph
       if y - p.height < 50:
           c.showPage()  # Start a new page
           y = 750  # Reset the y position to the top of the new page


       p.drawOn(c, 100, y-p.height)
       y -= p.height + spacing  # Adjust the spacing between paragraphs
       index += 1


   # Save the canvas as the final PDF
   c.save()


   # Save the PDF file into MongoDB
   with open(output_file, 'rb') as pdf_file:
       pdf_data = pdf_file.read()


   username = session.get('username')  # Get the username from session or any relevant source


   user = users_collection.find_one({'username': username})  # Retrieve the user document from MongoDB


   if user:
       user_id = user['_id']  # Assuming the user ID is stored in the '_id' field
       timestamp = datetime.datetime.now()  # Generate a timestamp


       file_name = session.get('file_name')


       pdf_document = {
           "user_id": user_id,
           "timestamp": timestamp,
           "file_name": file_name,
           "pdf_file": pdf_data,
           "subject_name": subject_name
       }


       pdf_collection.insert_one(pdf_document)
       print("PDF saved to MongoDB successfully.")
   else:
       print("User not found. PDF not saved.")






@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        subjects = []
        existing_user = users_collection.find_one({'username': username})

        if existing_user:
            return "Username already exists!"

        user = {'username': username, 'password': password, 'subjects': subjects}
        users_collection.insert_one(user)
        session['username'] = username
        return redirect('/')
    else:
        return render_template('register.html')


@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = users_collection.find_one({'username': username, 'password': password})

        if existing_user:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect('/userdashboard')
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
            return redirect('/')
    else:
        return render_template('login.html')




@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect('/')


@app.route("/userdashboard")
def user_dashboard():
    username = session.get('username')
    user = users_collection.find_one({'username': username})
    subjects=user['subjects']
    return render_template('dashboard.html',username=username,subjects=subjects)


import os
@app.route('/upload/<subject_name>', methods=['GET', 'POST'])
def upload(subject_name):
   if request.method == 'POST':
       file = request.files['file']
       if file:
           # Save the uploaded file to a temporary directory
           temp_dir = '/tmp'
           file_path = os.path.join(temp_dir, file.filename)
           file.save(file_path)
           session['file_name'] = file.filename
           # Perform text extraction
           extracted_text = extract_text_from_pdf(file_path)
           # Delete the temporary file
           os.remove(file_path)
           # Continue with text processing
           preprocessed_text = preprocess_text(extracted_text, segment_length=1700)
           print(len(preprocessed_text))
           summarized=summarizer(preprocessed_text,summary_model,summary_tokenizer)
           print(len(summarized))
           #keyword
           imp_keywords=extract_keywords_rake(extracted_text)
           questions = hf_run_model(preprocessed_text, num_return_sequences=8, num_questions=2)
           session['my_list'] = questions


           for count, ele in enumerate(questions):
               print(count + 1)
               print(ele)
         
           return render_template('upload.html', file_name=file.filename, text1=extracted_text,
                                  text2=preprocessed_text, text3=summarized, text4=imp_keywords, text5=questions,subject_name=subject_name)


   return render_template('upload.html',subject_name=subject_name)








@app.route('/generate_pdf/<subject_name>', methods=['GET'])
def generate_pdf(subject_name):
   items_1=session['my_list']
   output_path='output.pdf'
   convert_list_to_pdf_with_template(items_1,output_path,subject_name=subject_name)
   return redirect(url_for('my_pdf_documents', subject_name=subject_name))


def fetch_pdf_documents_for_user(username,subject_name):
   user = users_collection.find_one({'username': username})  # Retrieve the user document from MongoDB
   if user:
       user_id = user['_id']  # Assuming the user ID is stored in the '_id' field
       pdf_documents = pdf_collection.find({'user_id': user_id,'subject_name': subject_name})  # Fetch all PDF documents for the user
       return pdf_documents
   else:
       return None




@app.route('/view_pdf/<pdf_id>')
def view_pdf(pdf_id):
   pdf_doc = pdf_collection.find_one({'_id': ObjectId(pdf_id)})


   if pdf_doc:
       pdf_file = pdf_doc['pdf_file']
       pdf_buffer = io.BytesIO(pdf_file)
       return send_file(pdf_buffer,mimetype='application/pdf')


   return "PDF not found"


@app.route('/my_pdf_documents/<subject_name>/')
def my_pdf_documents(subject_name):
   username = session['username']  # Get the username from session or any relevant source
   if username:
       pdf_documents = fetch_pdf_documents_for_user(username,subject_name)
       return render_template('my_pdf_documents.html', pdf_documents=pdf_documents,subject_name=subject_name)

   return "User not logged in"


@app.route('/remove_pdf/<pdf_id>/<subject_name>', methods=['POST'])
def remove_pdf(pdf_id,subject_name):
   # Remove the PDF file from the database
   pdf_collection.delete_one({'_id': ObjectId(pdf_id)})
   return redirect(url_for('my_pdf_documents',subject_name=subject_name))


@app.route("/add_subject", methods=["POST"])
def add_subject():
    if request.method == "POST":
        subject_name = request.form.get("subjectName")
        username = session.get('username')
        user =users_collection.find_one({'username': username})

        if user:
            # Update the subjects list
            users_collection.update_one(
                {'username': session.get("username")},
                {'$push': {'subjects': subject_name}}
            )

            flash(f"Subject '{subject_name}' added successfully!", "success")
            return redirect(url_for("user_dashboard"))

    flash("Error adding subject", "danger")
    return redirect(url_for("user_dashboard"))


@app.route("/remove_subject", methods=["POST"])
def remove_subject():
    if request.method == "POST":
        subject_name = request.form.get("subject_name")
        username = session.get('username')
        user = users_collection.find_one({'username': username})

        if user:
            # Update the subjects list by pulling the specified subject
            users_collection.update_one(
                {'username': session.get("username")},
                {'$pull': {'subjects': subject_name}}
            )

            flash(f"Subject '{subject_name}' removed successfully!", "success")
            return redirect(url_for("user_dashboard"))

    flash("Error removing subject", "danger")
    return redirect(url_for("user_dashboard"))

