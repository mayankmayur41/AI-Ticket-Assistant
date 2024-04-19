import pandas as pd
import re
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import base64
from textblob import TextBlob
import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Load dataset and perform ETL process
def load_and_preprocess_data(dataset_path):
    data = pd.read_csv(dataset_path)
    # Perform data cleaning and preprocessing
    data['Document'] = data['Document'].apply(preprocess_text)
    return data

# Train machine learning model
def train_model(X_train, y_train):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    model = LogisticRegression(solver='saga', max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model, tfidf_vectorizer

# Predict topic_group based on ticket description
def predict_category(model, description, vectorizer):
    description = preprocess_text(description)
    description_vectorized = vectorizer.transform([description])
    category = model.predict(description_vectorized)[0]
    return category

# Preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Perform sentiment analysis
def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

# Perform entity recognition
def perform_entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Create ticket in ServiceNow
def create_servicenow_ticket(short_description, description, category, Impact, instance_url, username, password):
    url = f'{instance_url}/api/now/table/incident'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Basic ' + base64.b64encode(f'{username}:{password}'.encode()).decode()
    }
     # Define mapping between topic groups and assignment groups/services
    assignment_groups = {
        'Access': {'assignment_group': 'Help Desk', 'service': 'Electronic Messaging'},
        'Administrative rights': {'assignment_group': 'Change Management', 'service': 'IT Services'},
        'HR Support': {'assignment_group': 'IT Finance CAB', 'service': 'SAP Human Resources'},
        'Hardware': {'assignment_group': 'Hardware', 'service': 'Retail POS (Point of Sale)'},
        'Internal Project': {'assignment_group': 'Application Development', 'service': 'IT Services'},
        'Miscellaneous': {'assignment_group': 'Help Desk', 'service': 'IT Services'},
        'Purchase': {'assignment_group': 'Business Application Registration Approval Group', 'service': 'Retail Client Registration'},
        'Storage': {'assignment_group': 'Database', 'service': 'SAP Plant Maintenance'}
        
    }

    # Check if the predicted category is in the assignment groups mapping
    if category in assignment_groups:
        assignment_group = assignment_groups[category]['assignment_group']
        service = assignment_groups[category]['service']
    else:
        # Default assignment group and service if category not found in mapping
        assignment_group = 'Help Desk'  # Default to Help Desk for unspecified categories
        service = 'IT Services'  # Default service

    # Create incident data
    data = {
        'short_description': short_description,
        'description': description,
        'category': category,
        'Impact': Impact,
        'assignment_group': assignment_group,
        'service': service
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        ticket_details = response.json().get('result', {})
        ticket_number = ticket_details.get('number')
        print("Ticket created successfully in ServiceNow!")
        print("Ticket Number:", ticket_number)
        return ticket_details  # Return ticket details
    except requests.exceptions.RequestException as e:
        print("Failed to create ticket in ServiceNow:", e)
        return None
    

def main():
    # Step 1: ETL Process and Model Development
    dataset_path = r'C:\Users\61702\Documents\Designathon\\sample_dataset.csv'
    instance_url = 'https://dev193915.service-now.com'
    username = 'admin'
    password = 'Vj=Kw4l1oZI^'

    # Load and preprocess data
    data = load_and_preprocess_data(dataset_path)
    X = data['Document']
    y = data['Topic_group']
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train machine learning model
    model, tfidf_vectorizer = train_model(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(tfidf_vectorizer.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    
    # Step 2: Ticket Generation and ServiceNow Integration
    short_description = input("Enter a short description for the IT ticket: ")
    description = input("Enter a detailed description for the IT ticket: ")
    Impact = input("Enter the Impact for the IT ticket (e.g., High, Medium, Low): ")

    # Perform sentiment analysis
    sentiment = perform_sentiment_analysis(description)
    print("Sentiment Analysis:")
    print("Polarity:", sentiment.polarity)
    print("Subjectivity:", sentiment.subjectivity)

    # Perform entity recognition
    entities = perform_entity_recognition(description)
    print("Entity Recognition:")
    for entity, label in entities:
        print(f"Entity: {entity}, Label: {label}")

    
    predicted_category = predict_category(model, description, tfidf_vectorizer)
    
    print("Predicted Category:", predicted_category)
  
    
     # Create the ticket
    ticket_details = create_servicenow_ticket(short_description, description, predicted_category, Impact, instance_url, username, password)
    
    # Print selected ticket details if ticket was created successfully
    if ticket_details:
        print("\nTicket Details:")
        selected_items = {
            'short_description': ticket_details.get('short_description'),
            'description': ticket_details.get('description'),
            'category': ticket_details.get('category'),
            'Impact': ticket_details.get('Impact'),
        }
        for key, value in selected_items.items():
            print(f"{key}: {value}")
if __name__ == "__main__":
    main()




