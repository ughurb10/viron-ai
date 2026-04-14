import os
from flask import Flask, render_template, request, redirect, session, send_file, make_response, flash, url_for
import joblib
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import io
import json
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Assessment, UserProfile

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "vitalscope-secret")

# --- START OF THE FIXED DATABASE CONFIGURATION BLOCK ---
database_url = os.environ.get("DATABASE_URL")

# DEBUGGING: Check what DATABASE_URL is being read from environment
print(f"DEBUG: DATABASE_URL environment variable is: '{database_url}'")

if database_url and database_url.strip():
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    print(f"DEBUG: SQLALCHEMY_DATABASE_URI set to: '{database_url}' (from env)")
else:
    # Use SQLite for local development if DATABASE_URL is not set or empty
    sqlite_uri = "sqlite:///site.db"
    app.config["SQLALCHEMY_DATABASE_URI"] = sqlite_uri
    print(f"DEBUG: SQLALCHEMY_DATABASE_URI set to: '{sqlite_uri}' (default SQLite)")

app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database with the app
db.init_app(app)

# --- END OF THE FIXED DATABASE CONFIGURATION BLOCK ---


# Create tables
with app.app_context():
    db.create_all()

# Load all models
blood_model = joblib.load("disease_model.pkl")
cardio_model = joblib.load("cardio_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")

# Insights functions per dataset
def blood_insights(features):
    notes = []
    age, bp, glucose, hemoglobin, gender, rbc, mcv, mch = features

    if bp > 140:
        notes.append("High blood pressure detected.")
    elif bp < 90:
        notes.append("Low blood pressure detected.")

    if glucose > 125:
        notes.append("Elevated glucose levels, possible diabetes risk.")
    if hemoglobin < 12:
        notes.append("Low hemoglobin, possible anemia.")
    if rbc < 4.5:
        notes.append("Low RBC — could indicate anemia or other conditions.")
    if mcv < 80:
        notes.append("Low MCV — suggests microcytic red blood cells.")
    if mch < 27:
        notes.append("Low MCH — may reflect low hemoglobin per red blood cell.")

    return notes

def cardio_insights(features):
    notes = []
    age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active = features

    if ap_hi > 140 or ap_lo > 90:
        notes.append("High blood pressure readings detected.")
    if cholesterol > 1:
        notes.append("Elevated cholesterol levels.")
    if gluc > 1:
        notes.append("Elevated glucose levels.")
    if smoke == 1:
        notes.append("Patient smokes — risk factor.")
    if alco == 1:
        notes.append("Patient consumes alcohol regularly.")
    if active == 0:
        notes.append("Physical inactivity noted.")

    return notes

def diabetes_insights(features):
    # Placeholder: add custom insights if desired
    return []

def heart_insights(features):
    # Placeholder: add custom insights if desired
    return []

def generate_pdf_report(prediction, notes, form_type, form_data):
    """Generate a PDF health report"""
    # Create a BytesIO buffer to hold the PDF
    buffer = io.BytesIO()

    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#007bff'),
        alignment=1 # Center alignment
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#0056b3')
    )

    # Add title
    story.append(Paragraph("VitalScope Health Assessment Report", title_style))
    story.append(Spacer(1, 20))

    # Add report metadata
    report_info = [
        ["Report Date:", datetime.now().strftime("%B %d, %Y at %I:%M %p")],
        ["Assessment Type:", form_type.replace('_', ' ').title()],
        ["Report ID:", f"VS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"]
    ]

    info_table = Table(report_info, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(info_table)
    story.append(Spacer(1, 30))

    # Add assessment result
    story.append(Paragraph("Assessment Result", heading_style))

    # Determine result color based on prediction
    result_color = colors.green
    if any(word in prediction.lower() for word in ['high', 'risk', 'diabetes', 'anemia', 'hypertension']):
        if 'high' in prediction.lower():
            result_color = colors.red
        else:
            result_color = colors.orange

    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=result_color,
        spaceAfter=20,
        fontName='Helvetica-Bold'
    )

    story.append(Paragraph(f"<b>Result:</b> {prediction}", result_style))
    story.append(Spacer(1, 20))

    # Add input data summary
    story.append(Paragraph("Input Data Summary", heading_style))

    # Create data table based on form type
    if form_type == "blood":
        data_rows = [
            ["Parameter", "Value", "Reference Range"],
            ["Age", f"{form_data.get('age', 'N/A')} years", "18-120"],
            ["Blood Pressure", f"{form_data.get('bp', 'N/A')} mmHg", "90-140"],
            ["Glucose", f"{form_data.get('glucose', 'N/A')} mg/dL", "70-125"],
            ["Hemoglobin", f"{form_data.get('hemoglobin', 'N/A')} g/dL", "12-18"],
            ["Gender", "Female" if form_data.get('gender') == '0' else "Male", ""],
            ["RBC Count", f"{form_data.get('rbc', 'N/A')} million/μL", "4.0-5.5"],
            ["MCV", f"{form_data.get('mcv', 'N/A')} fL", "80-100"],
            ["MCH", f"{form_data.get('mch', 'N/A')} pg", "27-32"]
        ]
    elif form_type == "heart":
        data_rows = [
            ["Parameter", "Value", "Notes"],
            ["Age", f"{form_data.get('age', 'N/A')} years", ""],
            ["Sex", "Female" if form_data.get('sex') == '0' else "Male", ""],
            ["Chest Pain Type", form_data.get('cp', 'N/A'), ""],
            ["Resting BP", f"{form_data.get('trestbps', 'N/A')} mmHg", ""],
            ["Cholesterol", f"{form_data.get('chol', 'N/A')} mg/dL", ""],
            ["Fasting Blood Sugar", ">120 mg/dL" if form_data.get('fbs') == '1' else "≤120 mg/dL", ""],
            ["Max Heart Rate", f"{form_data.get('thalach', 'N/A')} bpm", ""],
            ["Exercise Angina", "Yes" if form_data.get('exang') == '1' else "No", ""]
        ]
    elif form_type == "diabetes":
        data_rows = [
            ["Parameter", "Value", "Reference Range"],
            ["Pregnancies", form_data.get('pregnancies', 'N/A'), ""],
            ["Glucose", f"{form_data.get('glucose', 'N/A')} mg/dL", "70-125"],
            ["Blood Pressure", f"{form_data.get('blood_pressure', 'N/A')} mmHg", "90-140"],
            ["Skin Thickness", f"{form_data.get('skin_thickness', 'N/A')} mm", ""],
            ["Insulin", f"{form_data.get('insulin', 'N/A')} μU/mL", ""],
            ["BMI", form_data.get('bmi', 'N/A'), "18.5-24.9"],
            ["Age", f"{form_data.get('age', 'N/A')} years", ""]
        ]
    else: # cardio
        data_rows = [
            ["Parameter", "Value", "Reference Range"],
            ["Age", f"{form_data.get('age', 'N/A')} years", ""],
            ["Gender", "Female" if form_data.get('gender') == '0' else "Male", ""],
            ["Height", f"{form_data.get('height', 'N/A')} cm", ""],
            ["Weight", f"{form_data.get('weight', 'N/A')} kg", ""],
            ["Systolic BP", f"{form_data.get('ap_hi', 'N/A')} mmHg", "90-140"],
            ["Diastolic BP", f"{form_data.get('ap_lo', 'N/A')} mmHg", "60-90"],
            ["Smoking", "Yes" if form_data.get('smoke') == '1' else "No", ""],
            ["Physical Activity", "Yes" if form_data.get('active') == '1' else "No", ""]
        ]

    data_table = Table(data_rows, colWidths=[2*inch, 1.5*inch, 2*inch])
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#007bff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    story.append(data_table)
    story.append(Spacer(1, 30))

    # Add health insights if available
    if notes:
        story.append(Paragraph("Health Insights & Recommendations", heading_style))
        for i, note in enumerate(notes, 1):
            story.append(Paragraph(f"{i}. {note}", styles['Normal']))
            story.append(Spacer(1, 8))
        story.append(Spacer(1, 20))

    # Add disclaimer
    story.append(Paragraph("Important Disclaimer", heading_style))
    disclaimer_text = """
    This report is generated by an AI-powered health assessment tool and is intended for informational purposes only.
    It should not be considered as a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider with any questions you may have
    regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of
    something you have read in this report.
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    story.append(Spacer(1, 20))

    # Add footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph("Generated by VitalScope AI Health Assistant", footer_style))

    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route("/", methods=["GET", "POST"])
def index():
    if not session.get("logged_in"):
        return redirect("/login")

    prediction = None
    notes = []
    error = None
    active_form = "blood" # Default visible form

    if request.method == "POST":
        form_type = request.form.get("form_type")
        active_form = form_type # keep active form after submission

        # Store form data in session for PDF generation
        session['last_form_data'] = dict(request.form)
        session['last_form_type'] = form_type

        try:
            if form_type == "blood":
                age = float(request.form["age"])
                bp = float(request.form["bp"])
                glucose = float(request.form["glucose"])
                hemoglobin = float(request.form["hemoglobin"])
                gender = float(request.form["gender"])
                rbc = float(request.form["rbc"])
                mcv = float(request.form["mcv"])
                mch = float(request.form["mch"])
                features = np.array([[age, bp, glucose, hemoglobin, gender, rbc, mcv, mch]])
                prediction = blood_model.predict(features)[0]
                notes = blood_insights([age, bp, glucose, hemoglobin, gender, rbc, mcv, mch])

            elif form_type == "cardio":
                age = float(request.form["age"])
                gender = float(request.form["gender"])
                height = float(request.form["height"])
                weight = float(request.form["weight"])
                ap_hi = float(request.form["ap_hi"])
                ap_lo = float(request.form["ap_lo"])
                cholesterol = int(request.form["cholesterol"])
                gluc = int(request.form["gluc"])
                smoke = int(request.form["smoke"])
                alco = int(request.form["alco"])
                active = int(request.form["active"])

                features = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
                prediction = cardio_model.predict(features)[0]
                notes = cardio_insights([age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active])

            elif form_type == "diabetes":
                pregnancies = float(request.form["pregnancies"])
                glucose = float(request.form["glucose"])
                blood_pressure = float(request.form["blood_pressure"])
                skin_thickness = float(request.form["skin_thickness"])
                insulin = float(request.form["insulin"])
                bmi = float(request.form["bmi"])
                diabetes_pedigree = float(request.form["diabetes_pedigree"])
                age = float(request.form["age"])

                features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
                prediction = diabetes_model.predict(features)[0]
                notes = diabetes_insights([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])

            elif form_type == "heart":
                age = float(request.form["age"])
                sex = int(request.form["sex"])
                cp = int(request.form["cp"])
                trestbps = float(request.form["trestbps"])
                chol = float(request.form["chol"])
                fbs = int(request.form["fbs"])
                restecg = int(request.form["restecg"])
                thalach = float(request.form["thalach"])
                exang = int(request.form["exang"])
                oldpeak = float(request.form["oldpeak"])
                slope = int(request.form["slope"])
                ca = int(request.form["ca"])
                thal = int(request.form["thal"])

                features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
                prediction = heart_model.predict(features)[0]
                notes = heart_insights([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

            # Store prediction and notes in session for PDF generation
            session['last_prediction'] = prediction
            session['last_notes'] = notes

            # Save assessment to database if user is logged in
            if session.get("user_id"):
                try:
                    assessment = Assessment(
                        user_id=session["user_id"],
                        assessment_type=form_type,
                        prediction=prediction
                    )
                    assessment.set_input_data(dict(request.form))
                    assessment.set_insights(notes)

                    db.session.add(assessment)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    # Continue even if saving fails

        except (ValueError, KeyError) as e:
            error = "Please fill in all fields correctly."

    return render_template("index.html", prediction=prediction, notes=notes, error=error, active_form=active_form)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Database authentication
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            session["logged_in"] = True
            session["user_id"] = user.id
            session["username"] = user.username
            flash(f"Welcome back, {username}!", "success")
            return redirect("/")
        else:
            error = "Invalid username or password."

    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            error = "Username already exists."
        elif User.query.filter_by(email=email).first():
            error = "Email already registered."
        else:
            # Create new user
            password_hash = generate_password_hash(password)
            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash
            )

            try:
                db.session.add(new_user)
                db.session.commit()

                # Log them in automatically
                session["logged_in"] = True
                session["user_id"] = new_user.id
                session["username"] = new_user.username
                flash(f"Welcome to VitalScope, {username}!", "success")
                return redirect("/")
            except Exception as e:
                db.session.rollback()
                error = "Registration failed. Please try again."

    return render_template("register.html", error=error)

@app.route("/history")
def history():
    if not session.get("logged_in"):
        return redirect("/login")

    user_id = session.get("user_id")
    if not user_id:
        return redirect("/login")

    # Get user's assessment history
    assessments = Assessment.query.filter_by(user_id=user_id).order_by(Assessment.created_at.desc()).limit(20).all()

    return render_template("history.html", assessments=assessments)

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    if not session.get("logged_in"):
        return redirect("/login")

    # Get data from session
    prediction = session.get('last_prediction')
    notes = session.get('last_notes', [])
    form_type = session.get('last_form_type')
    form_data = session.get('last_form_data', {})

    if not prediction or not form_type:
        return redirect("/")

    try:
        # Generate PDF
        pdf_buffer = generate_pdf_report(prediction, notes, form_type, form_data)

        # Create response
        response = make_response(pdf_buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=VitalScope_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

        return response

    except Exception as e:
        # If PDF generation fails, redirect back with error
        return redirect("/?error=pdf_generation_failed")

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("/login")

if __name__ == "__main__":
    # Ensure this runs in a development environment via 'flask run' or gunicorn for production
    # app.run(debug=True) # Recommended for local development only
    # For production, use gunicorn or similar WSGI server
    pass # Flask is usually run via 'flask run' command which handles debug itself.