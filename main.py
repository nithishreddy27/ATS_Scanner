import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File ,Form
from fastapi.responses import JSONResponse
import re
import os 
from fastapi.middleware.cors import CORSMiddleware
import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
from pdfrw import PdfReader, PdfDict, PdfObject
from pdf2image import convert_from_path
import cv2
import pytesseract

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
app = FastAPI()
# pytesseract.pytesseract.tesseract_cmd = "D:\\256SSD\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"



tool = language_tool_python.LanguageTool('en-US')
nlp = spacy.load("en_core_web_sm")




class NoTextFoundError(Exception):
    pass


parameters =  {}

headings_list = [
    "Contact Information", "Name", "Address", "Phone Number", "Email", "LinkedIn Profile", "Personal Website","PERSONAL INFO"
    "Summary", "Objective", "Professional Summary", "Career Objective", "Profile",
    "Experience", "Professional Experience", "Work Experience", "Employment History", "Relevant Experience",
    "Education", "Academic Background", "Qualifications", "Degrees",
    "Skills", "Technical Skills", "Skills", "Core Competencies", "Key Skills",
    "Certifications", "Licenses", "Qualifications", "Training",
    "Projects", "Key Projects", "Personal Projects", "Research Projects",
    "Achievements", "Awards", "Honors", "Recognitions",
    "Languages", "Language Proficiency", "Multilingual Skills",
    "Professional Affiliations", "Professional Associations", "Memberships", "Affiliations",
    "Volunteer Experience", "Community Service", "Volunteering",
    "Publications", "Articles", "Research Papers",
    "Interests", "Hobbies",
    "References", "Recommendations", "Referees",
    "Additional Information", "Extra Details", "Miscellaneous" ,"Patents"
]

desired_headings = [
    "Contact Information", "Summary", "Work Experience", "Education", "Skills", "Certifications", "Projects",
    "Achievements", "Languages", "Professional Affiliations", "Volunteer Experience", "Publications",
    "Interests", "References", "Additional Information" , "Awards",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your client's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def extract_text_from_pdf(pdf_path):
    def parse_content(contents):
        text = ""
        if isinstance(contents, list):
            for obj in contents:
                text += parse_content(obj)
        elif isinstance(contents, PdfObject):
            stream = contents.stream
            if stream:
                text += stream.decode('latin1', errors='ignore')
        elif isinstance(contents, PdfDict):
            if '/Contents' in contents:
                text += parse_content(contents.Contents)
        return text

    text = ''
    pdf = PdfReader(pdf_path)
    
    for page in pdf.pages:
        if '/Contents' in page:
            text += parse_content(page.Contents)

    return text


def preprocess_job_description(text):
    # Lowercase the text
    print("text ",text)
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    return filtered_words

def extract_keywords(tfidf_matrix, feature_names, top_n=5):
    keywords = []
    for row in tfidf_matrix:
        top_n_idx = np.argsort(row.toarray()).flatten()[-top_n:]
        keywords.append([feature_names[idx] for idx in top_n_idx])
    return keywords

def extract_job_keywords(text, top_n_keywords=5):
    data = {'description': [text]}
    # Load a pre-trained NER model
    nlp = spacy.load("en_core_web_sm")

    # Create a DataFrame from the provided data
    df = pd.DataFrame(data)
    
    # Function to extract named entities
    def extract_entities(text):
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "TECHNOLOGY", "WORK_OF_ART", "EVENT", "MONEY", "PERCENT"]]
        
        # Add custom entity extraction if needed
        custom_terms = ["Java", "JavaScript", "MS SQL", "jQuery", "HTML", "CSS"]
        for term in custom_terms:
            if term.lower() in text.lower():
                entities.append(term)
        
        return list(set(entities))  # Remove duplicates
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Function to extract top N keywords from TF-IDF matrix
    def extract_keywords(tfidf_matrix, feature_names, top_n=5):
        keywords = []
        for row in tfidf_matrix:
            top_n_idx = np.argsort(row.toarray()).flatten()[-top_n:]
            keywords.append([feature_names[idx] for idx in top_n_idx])
        return keywords
    
    # Extract keywords from TF-IDF matrix
    keywords = extract_keywords(tfidf_matrix, feature_names, top_n=top_n_keywords)
    
    # Process the document to extract entities
    df['entities'] = df['description'].apply(extract_entities)
    
    
    # Print the results
    for index, row in df.iterrows():
        combined_list = row['entities'] + keywords[index]
        combined_list = list(set(combined_list))  # Remove duplicates

        # print("\n=====Doc=====")
        # print(row['description'])
        # print("\n===Combined Entities and Keywords===")
        # print(combined_list)
    return combined_list

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # # # print("text ", text)
    # Remove unwanted characters (punctuation, special characters, etc.)
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # # # # print("cleaned text ",text)
    return text

def pos_tagging(text):
    # Tokenize the text into words
    print("inside pos ",text)
    try:
        words = word_tokenize(text)

         # Perform POS tagging
        pos_tags = pos_tag(words)
        # # # # print(pos_tags)
        return pos_tags
    except Exception as e:
        print("Error:", str(e))

def get_repeated_words(text):
   
    unwanted_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now' ,
    ])
    words = re.findall(r'\b\w+\b', text)

    # Remove unwanted words
    filtered_words = [word for word in words if word not in unwanted_words]

    # Count the frequency of each word
    word_counts = Counter(filtered_words)
    
    # Identify repeated words
    repeated_words = {word: count for word, count in word_counts.items() if count > 1}
    total_repeated_count = sum(repeated_words.values())
    max_repeated_count=50
    max_penalty=5
    penalty_score = max_penalty - (total_repeated_count / max_repeated_count * max_penalty)
    # Ensure the penalty score is not negative
    penalty_score = max(0, penalty_score)

    parameters['repeated_words'] = {
    'words': repeated_words,  # Initialize the list of words as empty
    'Total_number_of_repeated Words' :total_repeated_count,
    'score': None  # Initialize the score (can be replaced with a specific value)
    }
    
    return penalty_score

def get_action_verbs(text):
    doc = nlp(text)
    action_verbs = [token.text for token in doc if token.pos_ == "VERB"]
    # # # print("Action Verbs:", action_verbs)



    impactful_verbs = [
    "achieved", "accomplished", "attained", "exceeded", "surpassed", "generated", "delivered", "produced", "maximized",
    "led", "managed", "directed", "oversaw", "supervised", "motivated", "inspired", "developed", "mentor",
    "solved", "resolved", "analyzed", "identified", "improved", "optimized", "streamlined", "innovated", "transformed",
    "communicated", "presented", "negotiated", "collaborated", "persuaded", "influenced", "promoted", "coordinated", "facilitated",
    "designed", "implemented", "integrated", "engineered", "architected", "configured", "tested", "automated",
    "planned", "organized", "prioritized", "executed", "launched", "monitored", "evaluated", "completed",
    "increased", "reduced", "saved", "established", "expanded", "strengthened"
    ]

    weak_verbs = [
    "helped", "assisted", "supported", "tried", "attempted", "wanted", 
    "needed", "managed", "handled", "dealt with", "involved"
    ]
    weak_words = [
        "very", "really", "just", "somewhat", "maybe", "perhaps", "kind of", "sort of", 
        "a bit", "a little", "almost", "likely", "seems", "appears", "basically", "actually"
    ]
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in weak_verbs + weak_words) + r')\b'

# Find all weak verbs/words in the resume text
    found_weak_words = re.findall(pattern, text, flags=re.IGNORECASE)
    action_verb_count = sum(1 for token in doc if token.text.lower() in impactful_verbs)
    total_words = len([token for token in doc if token.is_alpha])
    score = (action_verb_count / total_words) * 100

    # print(" action_verb_count ",action_verb_count , " total_words ",total_words)
    action_verbs_score = score / 10
    
    # Ensure score is between 0 and 10
    # score = max(0, min(10, score))
    # Calculate action verbs score
    # action_verbs_score = 0
    # if len(action_verbs) > 0:
    #     action_verbs_score = ((len(action_verbs) - len(found_weak_words)) / len(action_verbs)) * 10
    #     action_verbs_score = max(0, min(10, action_verbs_score))
    # else:
    #     action_verbs_score = 0
    parameters['weak_verbs'] = {
    'words': found_weak_words,  # Initialize the list of words as empty
    'score': None  # Initialize the score (can be replaced with a specific value)
    }


    return (action_verbs ,found_weak_words ,action_verbs_score) 

def verb_tenses_suggest_improvements(tense_counts):
    suggestions = []

    # Example suggestions based on counts
    if tense_counts.get('Incorrect Usage or Non-Verb Words', 0) > 0:
        suggestions.append("Review and correct non-verb words or incorrect usages.")

    if tense_counts.get('Present Participle/Gerund', 0) > 15:
        suggestions.append("Consider reducing the use of present participles/gerunds and replace them with strong action verbs.")

    if tense_counts.get('Base Form', 0) > 5:
        suggestions.append("Replace base forms with more descriptive action verbs.")

    if tense_counts.get('Past Participle', 0) > 5:
        suggestions.append("Ensure past participles are used correctly in the context of past achievements.")

    return suggestions

def get_resume_length_score(num_pages):
    # score  = (1 - (num_pages / 2))*10
    if(num_pages == 1):
        score = 10
    elif(num_pages == 2):
        score = 5
    else:
        score = 0
    parameters['resume_length'] = {
    'length': num_pages,  # Initialize the list of words as empty
    'score': None  # Initialize the score (can be replaced with a specific value)
    }
    return score

def calculate_verb_tenses_score(tense_data):
    # Define points for each tense type
    tense_points = {
        'Present Tense': 10,
        'Past Tense': 10,
        'Present Participle/Gerund': 5,
        'Base Form': 5,
        'Past Participle': 5,
        'Incorrect Usage or Non-Verb Words': -10
    }

    # Initialize counters
    tense_counts = {tense: 0 for tense in tense_points.keys()}
    incorrect_count = 0
    # Count occurrences of each tense
    for _, tense in tense_data:
        if tense in tense_counts:
            tense_counts[tense] += 1
        else:
            # Handle incorrect usage or non-verb words
            incorrect_count+=1
            tense_counts['Incorrect Usage or Non-Verb Words'] += 1
    
    # Calculate total score
    total_score = 0
    total_verbs = sum(tense_counts.values())  # Total number of verbs
    if total_verbs == 0:
        return 0, tense_counts  # Avoid division by zero
    
    score = ( 1 - (incorrect_count / total_verbs) ) * 100
    total_score = score / 10
    # Calculate score based on proper usage and consistency
    # for tense, count in tense_counts.items():
    #     total_score += count * tense_points[tense]
    
    # # Normalize score to be within a range of 0 to 10
    # max_score = total_verbs * max(tense_points.values())
    # normalized_score = (total_score / max_score) * 10
    
    # return max(0, min(10, normalized_score)), tense_counts
    return total_score, tense_counts

def get_verb_tenses(pos_tags):
    verb_tenses = []
    for word, pos in pos_tags:
        if pos.startswith('VB'):  # Verbs start with 'VB' in POS tags
            tense = None
            if pos == 'VBD':
                tense = 'Past Tense'
            elif pos == 'VBG':
                tense = 'Present Participle/Gerund'
            elif pos == 'VBN':
                tense = 'Past Participle'
            elif pos == 'VBP':
                tense = 'Present Tense'
            elif pos == 'VBZ':
                tense = 'Present Tense (3rd person singular)'
            elif pos == 'VB':
                tense = 'Base Form'
            verb_tenses.append((word, tense))
    verb_tenses_score, tense_counts = calculate_verb_tenses_score(verb_tenses)
    improvement_suggestions = verb_tenses_suggest_improvements(tense_counts)

    parameters['verb_tenses'] = {
    'verb_tenses': verb_tenses,  # Initialize the list of words as empty
    'suggestions':improvement_suggestions ,
    'score': None  # Initialize the score (can be replaced with a specific value)
    }

    return verb_tenses_score ,improvement_suggestions

def count_numerical_characters(text):
    # Find all numerical occurrences in the text
    numerical_matches = re.findall(r'\b\d+(\.\d+)?%?\b', text)
    numerical_count = len(numerical_matches)

    # Determine if the numerical data is used effectively
    # For example, assign more points if numerical data is used in critical sections
    effective_numerical_score = 0
    if numerical_count > 0:
        effective_numerical_score = min(numerical_count / 10, 5)  # Adjust the scaling as needed


    parameters['numerical_characters'] = {
    'numerical_characters_count': numerical_count,  # Initialize the list of words as empty
    'score': None  # Initialize the score (can be replaced with a specific value)
}

    return numerical_count, effective_numerical_score

def find_headings_in_text(text, headings):
    found_headings = []
    for heading in headings:
        pattern = re.compile(rf'\b{re.escape(heading)}\b', re.IGNORECASE)
        if pattern.search(text):
            found_headings.append(heading)
    # # # # print("headings ",found_headings)
    parameters['headings'] = {
    'headings_found': found_headings,  # Initialize the list of words as empty
    'score': None  # Initialize the score (can be replaced with a specific value)
}

    return found_headings


def contains_experience_or_education(headings_used):
    # Define lists of keywords for experience and education
    experience_keywords = [
        "Experience", "Professional Experience", "Work Experience", "Employment History", "Relevant Experience"
    ]
    education_keywords = [
        "Education", "Academic Background", "Qualifications", "Degrees"
    ]
    
    # Check if any heading in headings_used contains an experience-related or education-related keyword
    has_experience = any(keyword in headings_used for keyword in experience_keywords)
    has_education = any(keyword in headings_used for keyword in education_keywords)
    
    return has_experience or has_education

def get_filler_wrods_and_count(text):
    # # print("inside fikller")
    filler_words = { "um", "uh", "like", "you know", "so", "very", "really", "basically",
    "just", "actually", "well", "seriously", "literally", "I mean", "kind of",
    "sort of", "to be honest", "in my opinion", "anyway", "more or less",
    "in general", "at the end of the day", "in essence", "in a nutshell",
    "to some extent", "as a matter of fact", "due to the fact that",
    "in the process of", "in the event that", "as such", "for example",
    "in terms of", "with regard to", "like I said", "pretty much", "a little bit",
    "at this point in time", "basically speaking", "in fact", "right now",
    "just about", "primarily", "in reality", "to some degree", "in the meantime",
    "at the end of the day", "you might say", "to tell you the truth",
    "in a sense", "I guess", "if you will", "essentially", "more or less",
    "at the moment", "in the long run", "with that being said", "in short",
    "with all due respect", "as a rule", "of course", "that being said",
    "on the whole", "in general terms", "as it turns out", "to be fair",
    "at the very least", "in particular", "as previously mentioned", "as mentioned earlier",
    "over time", "in a way", "basically speaking", "for the most part",
    "that is to say", "in the grand scheme of things", "if I'm being honest",
    "to be honest with you", "to put it another way", "if you ask me",
    "on the other hand", "from my perspective", "as it happens", "by and large",
    "as far as I know", "in light of", "it could be said that", "to be clear",
    "in the end", "speaking of which", "on balance", "as such",
    "with respect to", "without a doubt", "in effect", "to a certain extent",
    "to my knowledge", "considering that", "in summary", "at this stage",
    "in any case", "in other words", "by and large", "as it stands",
    "in practice", "with this in mind",
    "absolutely", "actually", "all in all", "at this time", "by no means",
    "due to", "effectively", "essentially", "fairly", "for that matter",
    "generally", "hence", "in actuality", "in addition", "in alignment with",
    "in a certain way", "in a general sense", "in an effort to", "in conjunction with",
    "in relation to", "in the absence of", "in the context of", "in the light of",
    "in the midst of", "in the process of", "in view of", "involving", "it should be noted",
    "just so", "more or less", "mostly", "notably", "obviously", "on occasion",
    "one might say", "perhaps", "presumably", "quite", "really", "relatively",
    "significantly", "simply", "so to speak", "strictly speaking", "to a large extent",
    "to an extent", "to a certain extent", "ultimately", "with respect to", "as part of",
    "as such", "by definition", "by way of", "considerably", "consequently",
    "effectively speaking", "for example", "for instance", "given that", "honestly",
    "in a nutshell", "in brief", "in this regard", "in this case", "it seems",
    "just about", "largely", "meanwhile", "more specifically", "not only",
    "obviously", "overall", "primarily", "significantly", "so to speak", "strictly",
    "substantially", "that is", "to clarify", "to illustrate", "to put it another way",
    "to summarize", "with that said", "as a result", "as a consequence", "essentially speaking",
    "for the most part", "in any event", "in the meantime", "it appears", "just in case",
    "largely speaking", "mostly speaking", "on balance", "partially", "specifically",
    "thus far", "ultimately speaking", "vis-à-vis", "when it comes to", "with reference to",
    "with respect to"}
    words = re.findall(r'\b\w+\b', text)
    
    # Find filler words in the text
    found_filler_words = [word for word in words if word in filler_words]
    filler_word_count = Counter(found_filler_words)
    total_filler_word_count = sum(filler_word_count.values())
    # print("filler_word_count ",found_filler_words)
    if words == 0:
        return 0
 
    filler_score = ( 1 - (total_filler_word_count/ len(words))) * 100
    score = filler_score / 20
    # Calculate the filler words score
    # filler_words_score = (1 - (total_filler_word_count / len(words))) * 5
    # score = max(0, min(5, filler_words_score))
    # Ensure the score is within the range [0, 5]

    parameters['filler_words'] = {
    'words': found_filler_words,  # Initialize the list of words as empty
    'count':filler_word_count,
    'score': None  # Initialize the score (can be replaced with a specific value)
}

    return (found_filler_words ,filler_word_count ,score)

def get_buzz_words_and_count(text):
    buzzwords_cliches = [
    "Synergy", "Leverage", "Paradigm shift", "Innovative", "Dynamic", "Results-driven",
    "Strategic", "Cutting-edge", "Proactive", "Forward-thinking", "Visionary", "Game-changer",
    "Impactful", "Best-in-class", "Agile", "High-performing", "Market-leading", "Out-of-the-box",
    "Revolutionary", "Holistic", "Groundbreaking", "Results-oriented", "Seamless", "Transformational",
    "Next-gen", "Disruptive", "Efficient", "Scalable", "Empower", "Trailblazing", "Industry-leading",
    "Customer-centric", "Strategic alignment", "Cutting-edge solutions", "Thought leader", "Metrics-driven",
    "Visionary leader", "Proactive approach", "Value-added", "High-impact", "Best practices", "Key driver",
    "High-value", "Targeted", "Leading-edge", "Synergistic", "Value-driven", "Growth-oriented", "Influencer",
    "Engaging", "Action-oriented", "World-class", "Frontline", "Advanced", "Core competencies", "Game-changing",
    "Competitive advantage", "Superior", "Trendsetting", "World-renowned", "Pinnacle", "Innovative solutions",
    "Critical thinker", "Breakthrough", "Mission-critical", "Top-tier", "End-to-end", "Benchmarked",
    "Big-picture", "Cutting-edge technology", "Cross-functional", "Deliverables", "Disruptive innovation",
    "End-to-end solutions", "High-level", "Influential", "Lean", "Milestone", "Next-level", "Outcome-focused",
    "Outside-the-box", "Performance-driven", "Process-oriented", "Quantifiable results", "Rapidly evolving",
    "Results-focused", "Scalable solutions", "Strategic vision", "Top-notch", "Turnkey", "User-centric",
    "Value proposition", "Vision-driven", "Winning", "360-degree view", "Actionable insights", "Best-in-breed",
    "Client-focused", "Continuous improvement", "Efficiency-driven", "Exceptional", "Forward-looking",
    "Growth-driven", "High-caliber", "High-impact results", "Integrated solutions", "Industry-standard",
    "Leading-edge technology", "Optimized", "Proven track record", "Reputable", "State-of-the-art",
    "Strategic initiatives", "Transformative", "Trendsetting solutions", "Value-added services",
    "Vibrant", "Well-versed", "Accomplished", "Advanced solutions", "Balanced", "Competitive", "Distinguished",
    "Elevated", "Exceptional performance", "Innovative strategies", "Leading practices", "Market-driven",
    "Next-generation", "Outstanding", "Productive", "Results-oriented strategies", "Risk-tolerant",
    "Strategic goals", "Sustainable", "Thought-provoking", "Top-performing", "User-focused", "Value-centric",
    "World-class service", "Agility", "Benchmarking", "Business-oriented", "Creative solutions",
    "Enhanced", "Expertise", "Groundbreaking solutions", "Impact-driven", "Milestone achievements",
    "Operational excellence", "Outstanding results", "Performance metrics", "Process improvement",
    "Strategic approach", "Targeted solutions", "Unique selling proposition", "Value creation",
    "Visionary thinking", "Adaptable", "Agile methodology", "Collaborative", "Customer-focused",
    "Expert-level", "Forward-thinking solutions", "Innovative techniques", "Market-leading solutions",
    "Performance optimization", "Proven results", "Reputable solutions", "Robust", "Strategic development",
    "Synergistic solutions", "Value-driven strategies", "Winning strategies", "Holistic approach",
    "Industry-best", "Key metrics", "Next-level strategies", "Outstanding performance", "Process excellence",
    "Strategic direction", "Trailblazing solutions", "User experience", "Value-enhanced", "World-class performance"
]

    words = re.findall(r'\b\w+\b', text)
    
    # Find filler words in the text
    found_buzz_words = [word for word in words if word in buzzwords_cliches]
    buzz_word_count = Counter(found_buzz_words)
    total_buzz_word_count = sum(buzz_word_count.values())

    # buzz_words_score = (total_buzz_word_count / len(words)) * 5
    # score = max(0, min(5, buzz_words_score))
    buzz_words_score = (1 - (total_buzz_word_count / len(words))) *100 
    score = buzz_words_score / 20


    parameters['buzz_words'] = {
    'words': found_buzz_words,  # Initialize the list of words as empty
    'count':buzz_word_count,
    'score': None  # Initialize the score (can be replaced with a specific value)
}
    return (found_buzz_words ,buzz_word_count ,score)

def get_dates_used_in_resume(text):
    date_patterns = {
    'month_year': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b',  # Full month name and year
    'month_year_abbr': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\b',  # Abbreviated month name and year
    'year_only': r'\b\d{4}\b',  # Year only
    'month_year_range': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\s-\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b',  # Month name range
    'month_year_range_abbr': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\s-\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\b',  # Abbreviated month range
    'year_range': r'\b\d{4}\s-\s\d{4}\b',  # Year range
}
    dates = []
    correct_dates = []
    incorrect_dates = []
    for key, pattern in date_patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            if key in ['month_year', 'month_year_abbr', 'year_only']:
                correct_dates.append(match)
            else:
                if re.match(date_patterns['month_year_range'], match):
                    correct_dates.append(match)
                else:
                    incorrect_dates.append({
                        'wrong_format': match,
                        'suggested_format': 'Month YYYY - Month YYYY' if 'range' in key else 'Month YYYY'
                    })

    # print("correct dates ",correct_dates , " in correct ",incorrect_dates )
    date_format_score = ( len(correct_dates) / (len(incorrect_dates) + len(correct_dates))) * 100
    score = date_format_score / 20
    # date_format_score = (1 - (len(incorrect_dates) / (len(incorrect_dates) + len(correct_dates)) )) * 10
    # score = max(0, min(10, date_format_score))
    parameters['dates'] = {
    'correct_dates': correct_dates,  # Initialize the list of words as empty
    'incorrect_dates':incorrect_dates,
    'score': None  # Initialize the score (can be replaced with a specific value)
}
    return correct_dates, incorrect_dates ,score
    # return correct_dates, incorrect_dates ,date_format_score
    
    # return (correct_dates ,unique_date_ranges)

def checkPresence(text):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # Regular expression for matching LinkedIn profile URLs
    linkedIn_regex = r'(https://www\.linkedin\.com/in/[a-zA-Z0-9_-]+|linkedin\.com/in/[a-zA-Z0-9_-]+)'
    
    # Search for email and LinkedIn profile in the text
    has_email = re.search(email_regex, text) is not None
    has_linkedIn = re.search(linkedIn_regex, text) is not None
    score = 0
    if(has_email and has_linkedIn ):
        score +=10
    elif(has_email or has_linkedIn):
        score+=5 
    else:
        score = 0
    return {
        'has_email': has_email,
        'has_linkedIn': has_linkedIn,
        'score':score
    }

def find_personal_pronouns(text):
    # Regular expression for matching personal pronouns
    pronoun_regex = r'\b(I|me|my|mine|we|us|our|ours|you|your|yours|he|him|his|she|her|hers|it|its|they|them|their|theirs)\b'
    words = re.findall(r'\b\w+\b', text)
    # Find all occurrences of personal pronouns in the text
    pronouns = re.findall(pronoun_regex, text, re.IGNORECASE)
    personal_pronouns_score = (1 - (len(pronouns) / len(words))) * 100
    score = personal_pronouns_score / 20
    # score = max(0, min(5, personal_pronouns_score))
    parameters['pronouns'] = {
    'words': words,  # Initialize the list of words as empty
    'score': None  # Initialize the score (can be replaced with a specific value)
}
    return pronouns ,score

def is_passive(sentence):
    # Parse the sentence using spaCy
    doc = nlp(sentence)
    
    # Check if the sentence contains any passive constructions
    for token in doc:
        if token.dep_ == "auxpass":  # "auxpass" indicates passive auxiliary verbs like "was", "is", etc.
            return True
    return False

def find_passive_sentences(bullet_points):
    passive_sentences = []
    
    for sentence in bullet_points:
        if is_passive(sentence):
            passive_sentences.append(sentence)

    # print("bullet points ",bullet_points)
    if len(bullet_points) > 0:
        passive_sentences_score = (1 - (len(passive_sentences) / len(bullet_points))) * 5
        # passive_score = (len(passive_sentences )/ len(bullet_points)) * 100
    else:
        passive_sentences_score = 5
        # passive_score = 5  # If there are no bullet points, you could default to a perfect score.

   
    score = passive_sentences_score / 20
    # score = max(0, min(5, passive_sentences_score))
    parameters['passive_sentences'] = {
    'passive_sentences': passive_sentences,  # Initialize the list of words as empty
    'score': None  # Initialize the score (can be replaced with a specific value)
}
    return passive_sentences ,score

def standardize_capitalization(sentence):
    if sentence:
        return sentence[0].upper() + sentence[1:].lower()
    return sentence

def standardize_spacing(sentence):
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', sentence)
    return sentence.strip()

def ensure_punctuation(sentence):
    if not sentence.endswith(('.', '?', '!')):
        return sentence + '.'
    return sentence

def standardize_quotes(sentence):
    sentence = sentence.replace('“', '"').replace('”', '"')
    sentence = sentence.replace("‘", "'").replace("’", "'")
    return sentence

def remove_leading_special_characters(sentence):
    match = re.match(r'^([^\w\s]+)', sentence)
    if match:
        return match.group(0), sentence[len(match.group(0)):]
    return '', sentence

def reapply_special_characters(special_chars, sentence):
    if(len(sentence) <= 2):
        return special_chars + sentence
    else:
        return special_chars +" "+ sentence

def standardize_sentence(sentence):
    special_chars, core_sentence = remove_leading_special_characters(sentence)
    # # # print("core_sentence ",core_sentence)
    if(len(core_sentence) > 0 ): 
        standardized_core_sentence = standardize_capitalization(core_sentence.strip())
        standardized_core_sentence = standardize_spacing(standardized_core_sentence)
        standardized_core_sentence = ensure_punctuation(standardized_core_sentence)
        standardized_core_sentence = standardize_quotes(standardized_core_sentence)
        return reapply_special_characters(special_chars, standardized_core_sentence)
    else:
        return reapply_special_characters(special_chars,core_sentence)

def check_sentence(sentence):
    standardized_sentence = standardize_sentence(sentence)
    if sentence != standardized_sentence:
        return {
            "your statement": sentence,
            "consistent statement": standardized_sentence
        }
    else:
        return {
            "your statement": sentence,
            "consistent statement": "Already consistent"
        }

def check_statements(statements):
    results = []
    for sentence in statements:
        result = check_sentence(sentence)
        if result['consistent statement'] != "Already consistent":
            results.append(result)
    
    if len(statements) > 0:
        consitency_score = ((len(statements) - len(results)) / len(statements)) * 100
    else:
        consitency_score = 100  # Default to a perfect score if there are no statements

    score = consitency_score / 10
    # score = max(0, min(10, cositency_score))
    
    parameters['consistency'] = {
        'results': results,
        'score': score
    }
    
    return results, score

def calculate_Heading_score(found_headings, desired_headings):
    # score = 0
    # for heading in found_headings:
    #     if heading in desired_headings:
    #         score += 1  # Award points if the heading is desired
    #     # else:
    #     #     score -= 1  # Deduct points if the heading is not desired
    # # # # # print("score ",score)
    # return score
    # # print("found headings ",found_headings , " desired headings ",desired_headings)
    score =( len(found_headings) / len(desired_headings)) * 100
    heading_score  = score / 20
    return heading_score

def get_bullet_points_and_non_bullet_points(text_positions):
    bullet_points = []
    non_bullet_points = []
    bullet_markers = ["•", "-", "*", "–"]  # Add any additional bullet markers

    for item in text_positions:
        text = item["text"].strip()
        lines = text.splitlines()

        for line in lines:
            stripped_line = line.strip()
            # Check if the line is a bullet point
            if stripped_line.startswith(tuple(bullet_markers)) or \
               (stripped_line.split()[0].strip(".").isdigit()) or \
               (stripped_line.split()[0].strip(".").isalpha() and len(stripped_line.split()[0].strip(".")) == 1):
                bullet_points.append(stripped_line)
            else:
                non_bullet_points.append(stripped_line)

    parameters['bullet_points'] = {
    'bullet_points': bullet_points,  # Initialize the list of words as empty
    'score': None  # Initialize the score (can be replaced with a specific value)
} 
    return bullet_points, non_bullet_points

def calculate_average_length_and_count(sentences):
    total_length = sum(len(sentence.split()) for sentence in sentences)
    count = len(sentences)
    average_length = total_length / count if count else 0
    return count, average_length

def get_counts_and_average_lengths(text_positions):
    # Get bullet points and non-bullet points
    bullet_points, non_bullet_points = get_bullet_points_and_non_bullet_points(text_positions)

    # Calculate counts and average lengths
    bullet_count, avg_bullet_length = calculate_average_length_and_count(bullet_points)
    non_bullet_count, avg_non_bullet_length = calculate_average_length_and_count(non_bullet_points)

    return bullet_count, avg_bullet_length, non_bullet_count, avg_non_bullet_length


def calculate_bullet_points_score(bullet_count):
  
    
    # Number of bullet points
    
    num_bullet_points = bullet_count
    # Calculate average length of bullet points
    
    # Example scoring for bullet points
    bullet_points_score = min(num_bullet_points / 10, 5)  # Adjust the scaling as needed
    # bullet_points_score = min(num_bullet_points / 10, 5)  # Adjust the scaling as needed
    
    return bullet_points_score 

def check_grammar(text):
    matches = tool.check(text)

    # Correct the text
    corrected_text = tool.correct(text)

    num_errors = len(matches)
    # # # print("num_errors ",num_errors)
    score = max(0, 100 - (num_errors * 1))
    parameters['grammar'] = {
    # 'errors':matches,
    'corrected_text': corrected_text,  # Initialize the list of words as empty
    'score': None  # Initialize the score (can be replaced with a specific value)
}
    return matches , corrected_text , score

def int_to_rgb(color_int):
    """
    Converts an integer representing a color to an RGB tuple.
    
    Parameters:
        color_int (int): The color integer.
    
    Returns:
        tuple: A tuple containing the RGB values (r, g, b).
    """
    b = color_int & 255
    g = (color_int >> 8) & 255
    r = (color_int >> 16) & 255
    return r, g, b

def extract_unique_fonts_sizes_colors_styles(text_positions):
    """
    Extracts a unique list of colors and styles from the text positions.
    
    Parameters:
        text_positions (list): A list of dictionaries containing text and its properties.
    
    Returns:
        dict: A dictionary with 'colors' and 'styles' as keys, containing unique RGB colors and styles respectively.
    """
    unique_fonts = set()
    unique_sizes = set()
    unique_colors = set()
    unique_styles = set()

    for item in text_positions:
        # Add font name and size to their respective sets
        unique_fonts.add(item['font'])
        unique_sizes.add(item['size'])

        # Convert color to RGB and add to set
        color_rgb = int_to_rgb(item['color'])
        unique_colors.add(color_rgb)

        # Add font style (flags) to set
        unique_styles.add(item['flags'])

    return {
        'fonts': list(unique_fonts),
        'sizes': sorted(list(unique_sizes)),  # Sorted list of font sizes
        'colors': list(unique_colors),
        'styles': list(unique_styles)
    }

def get_cosine_similarity(resume_text , job_description_text):
    job_text = job_description_text
    documents = [job_text, resume_text]
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity between the job description and resume summary
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    # print("cosine sime ",cosine_sim)

    cosine_sim_score = cosine_sim[[0]] * 10

    return cosine_sim_score

def extract_text_with_positions(pdf_path):
    # # # # print("iNside positions")
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    # print("pages ",num_pages )
    text_positions = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:  # Use "dict" to get detailed information
            if block['type'] == 0:  # Text blocks
                # print("inside block")
                for line in block['lines']:
                    for span in line['spans']:
                        text_positions.append({
                            'text': span['text'].strip(),
                            'bbox': span['bbox'],  # (x0, y0, x1, y1)
                            'font': span['font'],  # Font name
                            'size': span['size'],  # Font size
                            'color': span['color'],  # Font color in integer format
                            'flags': span['flags'],  # Font styles (e.g., bold, italic)
                        })
    doc.close()
    return (text_positions , num_pages)

def extract_text_from_positions(text_positions):
    text = ""
    for i in text_positions:
        text+=i["text"]+" "
    return text

def calculate_keyword_match_score(resume_text, job_description_text):
    filtered_job_words = preprocess_job_description(job_description_text)
    filtered_resume_words = preprocess_job_description(resume_text)
    job_keywords = extract_job_keywords(job_description_text )
    resume_keywords =  extract_job_keywords(resume_text)

    job_counter = Counter(job_keywords)
    resume_counter = Counter(resume_keywords)

    common_keywords = set(job_counter) & set(resume_counter)
    
    parameters['keywords'] = {
    'job_keywords': job_keywords,  # Initialize the list of words as empty
    'resume_keywords':resume_keywords,
    'score': None  # Initialize the score (can be replaced with a specific value)
}
    
    score = (len(common_keywords)/ len(filtered_resume_words)) * 100
    keyword_score = score / 5
    # if number_of_relevant_keywords == 0:
    #     keyword_optimization_score = 0
    #     return keyword_optimization_score   
    # else:
    #     keyword_optimization_score = (len(total_resume_keywords) / number_of_relevant_keywords) * 100
    #     # print("keyword_optimization_score ",keyword_optimization_score)
    #     return keyword_optimization_score / 5
  

    return keyword_score
    # # print("Extracted Keywords raka:", ext_filtered_words)

def normalize(value, min_value, max_value):
    if max_value == min_value:  # Avoid division by zero
        return 0
    return (value - min_value) / (max_value - min_value)


# def calculate_ats_score(numerical_count , heading_score ,bullet_count ,grammar_score , cosine_sim):
    

#     features = {
#         'numerical_count': numerical_count,
#         'headings_used': heading_score,
#         'number_of_points_used': bullet_count,
#         'corrected_grammar': grammar_score,
#         'keyword_matching': cosine_sim  # Cosine similarity used here
#     }

#     normalized_features = {
#         'numerical_count': normalize(features['numerical_count'], min_value=0, max_value=200),
#         'headings_used': normalize(features['headings_used'], min_value=0, max_value=10),
#         'number_of_points_used': normalize(features['number_of_points_used'], min_value=0, max_value=50),
#         'corrected_grammar': features['corrected_grammar'] / 100,  # Assuming it's out of 100
#         'keyword_matching': features['keyword_matching'][0][0]  # Assuming cosine similarity is already between 0 and 1
#     }

#     # # # # print("normalized_features ",normalized_features)
#     weights = {
#         'numerical_count': 0.15,
#         'headings_used': 0.10,
#         'number_of_points_used': 0.20,
#         'corrected_grammar': 0.25,
#         'keyword_matching': 0.30
#     }
#     # normalized_numerical_count = normalize(features['numerical_count'], min_value=0, max_value=20)
#     # normalized_headings_used = normalize(features['headings_used'], min_value=0, max_value=10)
#     # normalized_number_of_points_used = normalize(features['number_of_points_used'], min_value=0, max_value=15)
#     # normalized_corrected_grammar = features['corrected_grammar']  # Already normalized
    
#     # Cosine similarity for keyword matching
#     # normalized_keyword_matching = features['keyword_matching']
#     ats_score = (
#         normalized_features['numerical_count'] * weights['numerical_count'] +
#         normalized_features['headings_used'] * weights['headings_used'] +
#         normalized_features['number_of_points_used'] * weights['number_of_points_used'] +
#         normalized_features['corrected_grammar'] * weights['corrected_grammar'] +
#         normalized_features['keyword_matching'] * weights['keyword_matching']
#     )
#     # Scale the ATS score to 0-100
#     ats_score *= 100

#     # # # # print("ats_score ",ats_score)
#     return ats_score



def calculate_ats_score(
    action_verbs_score,
    verb_tenses_score,
    filler_words_score,
    buzz_words_score,
    dates_formatting_score,
    contact_information_score,
    personal_pronouns_score,
    passive_sentences_score,
    consistency_score,
    numerical_count_score,
    heading_score,
    grammar_score,
    cosine_similarity_score,
    resume_length_Score,
    keyword_optimization_score,
    bullet_point_score,
    headings_used
):
    # Define weights for each metric
    # Define weights for each metric
    weights = {
        'action_verbs': 0.10,
        'verb_tenses': 0.10,
        'filler_words': 0.05,
        'buzz_words': 0.05,
        'dates_formatting': 0.05,
        'contact_information': 0.10,
        'personal_pronouns': 0.05,
        'passive_sentences': 0.05,
        'consistency': 0.10,
        'numerical_count': 0.07,
        'heading': 0.05,
        'grammar': 0.10,
        'cosine_similarity': 0.12,
        'resume_length': 0.10,
        'repeated_count': -0.05,
        'keyword_optimization': 0.20,
        'bullet_points': 0.08,  # Add weight for bullet points
    }



    print("action_verbs_score ",action_verbs_score )
    print(" verb_tenses_score ",verb_tenses_score) 
    print(" filler_words_score ",filler_words_score) 
    print(" buzz_words_score ",buzz_words_score) 
    print(" dates_formatting_score ",dates_formatting_score) 
    print(" contact_information_score ",contact_information_score) 
    print(" personal_pronouns_score ",personal_pronouns_score) 
    print(" passive_sentences_score ",passive_sentences_score) 
    print(" consistency_score ",consistency_score) 
    print(" numerical_count_score ",numerical_count_score) 
    print(" heading_score ",heading_score) 
    print(" grammar_score ", grammar_score * weights['grammar']) 
    print(" cosine_similarity_score ",cosine_similarity_score)
    print(" resume_length_Score ",resume_length_Score ) 
    print(" bullet_points ",bullet_point_score  ) 
    print(" keyword_optimization_score ", keyword_optimization_score * weights['keyword_optimization'] )
    # Calculate the ATS score based on weights
 
    parameters['resume_length']['score'] = resume_length_Score * weights['resume_length'] 
    parameters['keywords']['score'] = keyword_optimization_score * weights['keyword_optimization'] 
    # parameters['cosine_similarity']['score'] = cosine_similarity_score * weights['cosine_similarity']
    parameters['grammar']['score'] = grammar_score * weights['grammar']
    parameters['bullet_points']['score'] = bullet_point_score * weights['bullet_points'] 
    parameters['consistency']['score'] = consistency_score * weights['consistency']
    parameters['passive_sentences']['score'] = passive_sentences_score * weights['passive_sentences']
    parameters['dates']['score'] = dates_formatting_score * weights['dates_formatting']
    parameters['buzz_words']['score'] = buzz_words_score * weights['buzz_words']
    parameters['filler_words']['score'] = filler_words_score * weights['filler_words']
    parameters['numerical_characters']['score'] = numerical_count_score * weights['numerical_count']
    parameters['verb_tenses']['score'] = verb_tenses_score * weights['verb_tenses']
    parameters['weak_verbs']['score'] = action_verbs_score * weights['action_verbs']
    parameters['pronouns']['score'] = personal_pronouns_score * weights['personal_pronouns']
    parameters['headings']['score'] = heading_score * weights['heading']
  

    ats_score = (
        (action_verbs_score ) +
        (verb_tenses_score ) +
        (filler_words_score ) +
        (buzz_words_score ) +
        (dates_formatting_score) +
        (contact_information_score ) +
        (personal_pronouns_score ) +
        (passive_sentences_score ) +
        (consistency_score ) +
        (numerical_count_score ) +
        (heading_score ) +
        (grammar_score * weights['grammar'] ) +
        (cosine_similarity_score )+
        (resume_length_Score ) +
        (keyword_optimization_score )+
        (bullet_point_score  )

    )

    ats_score = (ats_score * len(headings_used))/len(desired_headings)

    print("score ",ats_score)
    # ats_score = (
    #     (action_verbs_score * weights['action_verbs']) +
    #     (verb_tenses_score * weights['verb_tenses']) +
    #     (filler_words_score * weights['filler_words']) +
    #     (buzz_words_score * weights['buzz_words']) +
    #     (dates_formatting_score * weights['dates_formatting']) +
    #     (contact_information_score * weights['contact_information']) +
    #     (personal_pronouns_score * weights['personal_pronouns']) +
    #     (passive_sentences_score * weights['passive_sentences']) +
    #     (consistency_score * weights['consistency']) +
    #     (numerical_count_score * weights['numerical_count']) +
    #     (heading_score * weights['heading']) +
    #     (grammar_score * weights['grammar']) +
    #     (cosine_similarity_score * weights['cosine_similarity'])+
    #     (resume_length_Score * weights['resume_length'] ) +
    #     (keyword_optimization_score * weights['keyword_optimization'] )+
    #     (bullet_point_score * weights['bullet_points'] )

    # )
    ats_score = max(0, min(100, ats_score))
    # print("ats scrore ",ats_score)
    return ats_score


def extract_text_with_positions_new(pdf_path):
    text_positions = []
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    for character in text_line:
                        if isinstance(character, LTChar):
                            text_positions.append({
                                'text': character.get_text(),
                                'bbox': character.bbox,  # (x0, y0, x1, y1)
                                'fontname': character.fontname,
                                'size': character.size,
                            })
    return text_positions

@app.post("/upload_pdf/")
async def upload_pdf(
   file: UploadFile = File(...),
  job_description: str = Form(...) 
    ):
    pdf_path = f"./{file.filename}"
    print("new request came ",pdf_path)
    try:
        # Save the uploaded file
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
 
        # Extract text positions and draw rectangles
        print("PDF Input Done")
        text_positions , num_pages = extract_text_with_positions(pdf_path)


        print("Posititons done ",text_positions)
        aggregated_text = extract_text_from_positions(text_positions)

        if(aggregated_text):
            pass
        else:
            print("text not found using normal way")
            pages = convert_from_path(pdf_path)
            extracted_text = []
            aggregated_text = ""
            for page in pages:
                # Step 2: Preprocess the image (deskew)
                preprocessed_image = deskew(np.array(page))

                # Step 3: Extract text using OCR
                text = extract_text_from_image(preprocessed_image)
                extracted_text.append(text)

            # Print the extracted text from each page
            for i, text in enumerate(extracted_text):
                # print(f"Page {i+1} Text:\n{text}\n")
                aggregated_text += text
            print("aggregated_text ",aggregated_text)
        if(aggregated_text):
                    cleaned_text = preprocess_text(aggregated_text)
                    print("text cleaning done")
                    headings_used = find_headings_in_text(aggregated_text, headings_list)
                    print("Extracted headings")
                    
                    contain_experience = contains_experience_or_education(headings_used)
                    print("Experience checked")
                    if(contain_experience):
                        print("Experience found")
                        heading_score = calculate_Heading_score(headings_used, desired_headings)
                        print("Scoring headings ",heading_score)
                        repeated_count_score = get_repeated_words(cleaned_text)
                        print("scoring repeated words ",repeated_count_score)
                        # # print("repeated_count_score ",repeated_count_score)
                        pos_tags = pos_tagging(cleaned_text)
                        print("POs tagging")

                        action_verbs , found_weak_words , action_verbs_score = get_action_verbs(aggregated_text)
                        print("scoring action verbs ",action_verbs_score)


                        resume_length_Score = get_resume_length_score(num_pages)
                        # print("scoring resume length ",resume_length_Score)
                        # # # print("Action verbs ",action_verbs)
                        # # print("action_verbs_score ",action_verbs_score)
                        verb_tenses_score , improvement_suggestions = get_verb_tenses(pos_tags)
                        # print("scoring verb tenses ",verb_tenses_score)
                        # # print("verb_tenses ",verb_tenses_score ,improvement_suggestions)

                        found_filler_words , filler_word_count ,filler_words_score = get_filler_wrods_and_count(cleaned_text)
                        # print("scoring filler words ",filler_words_score)
                        # # print("filler_words_score ",filler_words_score)

                        found_buzz_words , buzz_word_count  ,buzz_word_score = get_buzz_words_and_count(cleaned_text)
                        # # print("scoring buzz words ",bullet_count_score)
                        # print("buzz_word_score ",buzz_word_score)
                        correct_dates, incorrect_dates ,date_format_score = get_dates_used_in_resume(aggregated_text)
                        # print("scoring dates ",date_format_score)

                        # # print("correct_dates ",date_format_score)
                        email_linkedIn =  checkPresence(aggregated_text)
                        # print("Scoring contact info")
                        contact_information_score = email_linkedIn["score"]
                        # print("Checking personal info ",contact_information_score)
                        # # print("contact_information_score   ",contact_information_score)
                        pronouns , pronouns_score= find_personal_pronouns(cleaned_text)
                        # print("Scoring propnouns ",pronouns_score)
                        # # print("personal pronouns   ",pronouns_score)

                        # print("fetching bullet points")
                        bullet_points, non_bullet_points = get_bullet_points_and_non_bullet_points(text_positions)
                        # # # print("personal pronouns used in resume  ",pronouns)

                        # passive_sentences = find_passive_sentences(bullet_points)
                        # print("Scoring passive sentences")

                        passive_sentences , passive_sentences_score= find_passive_sentences(bullet_points)
                        # print("passive_sentences in resume ",passive_sentences_score)
                        # print("Scoring conistency")


                        consistency_results ,consistency_score = check_statements(bullet_points)
                        # print("consistency of points in resume " ,consistency_score)
                        # # # # print("pronouns  ",passive_sentences)
                        # print("Scoring numerical ")
                        numerical_count ,numerical_score = count_numerical_characters(cleaned_text)
                        # print("numerical_count ", numerical_score)

                        # # print("heading_score  ",heading_score)
                        # # Cleanup files
                        # print("Scoring bullet points")
                        bullet_count, avg_bullet_length, non_bullet_count, avg_non_bullet_length = get_counts_and_average_lengths(text_positions)
                        bullet_count_score = calculate_bullet_points_score(bullet_count)
                        # print("total points used in resume ",bullet_count_score)
                        

                        # print("-"*10)
                        # print("Scoring grammar")
                        matches, corrected_text , grammar_score = check_grammar(aggregated_text)

                        # # print("Grammar Score ",grammar_score)
                        # print("extracting sizes and colors")
                        unique_data = extract_unique_fonts_sizes_colors_styles(text_positions)

                        # print("Getting cosine similarity")
                        cosine_sim = get_cosine_similarity(aggregated_text , job_description)

                        # print("Scoring key words ",cosine_sim)
                        # # print("cosine sim in ",cosine_sim[0][0])
                        keyword_match_score = calculate_keyword_match_score( cleaned_text , job_description)
                        # # print(f"Cosine Similarity between Job Description and Resume Summary: {cosine_sim[0][0]:.2f}")

                        # print("Scoring ATS ")
                        ats_score = calculate_ats_score(action_verbs_score=action_verbs_score , verb_tenses_score=verb_tenses_score , filler_words_score= filler_words_score , buzz_words_score= buzz_word_score , dates_formatting_score= date_format_score , personal_pronouns_score= pronouns_score , passive_sentences_score= passive_sentences_score ,consistency_score = consistency_score , numerical_count_score= numerical_score ,heading_score=heading_score , grammar_score= grammar_score ,cosine_similarity_score=cosine_sim[0][0] ,contact_information_score = contact_information_score ,resume_length_Score=resume_length_Score , keyword_optimization_score = keyword_match_score , bullet_point_score=bullet_count_score ,headings_used=headings_used )
                        # print("ats score ",ats_score)
                    else:
                        raise NoTextFoundError("No heading related to experience or education was found")

        else:
            raise NoTextFoundError("No text provided")

        
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        # # # print("Operations done returning")

        return JSONResponse(content={"text": aggregated_text , "TextPositions":text_positions , "Parameters":parameters})
        

    except Exception as e:
        # Ensure files are cleaned up on error
        # print("ex ",e)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/test_pdf/")
async def test_pdf(file: UploadFile = File(...)):
    pdf_path = f"./{file.filename}"
    # # # # print("new request came ",pdf_path)
    try:
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
        # Save the uploaded file

        extracted_text = extract_text_from_pdf(pdf_path)
        pos = extract_text_with_positions_new(pdf_path)
        # print(extracted_text , pos)
      
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        # # # print("Operations done returning")

        return JSONResponse(content={"text":""})
        
 
    except Exception as e:
        # Ensure files are cleaned up on error
        # print("ex ",e)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/test_pdf_rw/")
async def test_pdf(file: UploadFile = File(...)):
    pdf_path = f"./{file.filename}"
    # # # # print("new request came ",pdf_path)
    try:
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
        # Save the uploaded file

        extracted_text = extract_text_from_pdf(pdf_path)
       
        # print("text ",extracted_text)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        # # # print("Operations done returning")

        return JSONResponse(content={"text":""})
        

    except Exception as e:
        # Ensure files are cleaned up on error
        # print("ex ",e)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    