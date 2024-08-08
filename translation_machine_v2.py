import logging
from flask import Flask, jsonify, request
from flask_restful import Api, reqparse, Resource
from transformers import pipeline
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')

# Initialize Flask application
app = Flask(__name__)
api = Api(app)

# Setup basic logging configuration
logging.basicConfig(filename='machine_translation_v2.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize translation pipelines
translation_pipes = {
    'french': pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
    'english': pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
}

# Define expanded medical abbreviation dictionaries
med_abbr_en_to_fr = {
    'BP': 'TA',  # Blood Pressure
    'HR': 'FC',  # Heart Rate
    'ECG': 'ECG',  # Electrocardiogram (same abbreviation in French)
    'CT scan': 'Tomodensitogrammes',
    'CPR': 'RCR',
    'DVT': 'TVP',
    'GERD': 'RGO',
    'TKA': 'PTG',
    'MRI': 'IRM',  # Magnetic Resonance Imaging
    'IV': 'IV',  # Intravenous (same abbreviation in French)
    'CBC': 'NFS',  # Complete Blood Count
    'WBC': 'GB',  # White Blood Cells
    'RBC': 'GR',  # Red Blood Cells
    'UTI': 'ITU',  # Urinary Tract Infection
    'COPD': 'BPCO',  # Chronic Obstructive Pulmonary Disease
    'CHF': 'IC',  # Congestive Heart Failure
}

med_abbr_fr_to_en = {v: k for k, v in med_abbr_en_to_fr.items()}

def preprocess_text(text, direction='en_to_fr'):
    """Preprocess text by handling medical abbreviations."""
    abbr_dict = med_abbr_en_to_fr if direction == 'en_to_fr' else med_abbr_fr_to_en
    
    # Replace abbreviations with their full forms
    for abbr, full_form in abbr_dict.items():
        text = text.replace(abbr, full_form)
    
    return text

def postprocess_text(text, direction='en_to_fr'):
    """Restore medical abbreviations in the translated text."""
    abbr_dict = med_abbr_en_to_fr if direction == 'en_to_fr' else med_abbr_fr_to_en
    
    # Replace full forms with abbreviations
    for full_form, abbr in abbr_dict.items():
        text = text.replace(full_form, abbr)
    
    return text

def tokenize(text):
    """Tokenize text using NLTK."""
    return nltk.word_tokenize(text.lower())

def metrics(predicted, reference):
    """Compute METEOR and BLEU scores."""
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(predicted)
    
    if not reference_tokens or not candidate_tokens:
        return None, None
    
    # Calculate METEOR score
    meteor = meteor_score.meteor_score([reference_tokens], candidate_tokens)
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
    
    return meteor, bleu_score

def translate_french_to_english(french_text):
    """Translate French text to English and handle medical abbreviations."""
    # Preprocess the French text
    processed_text = preprocess_text(french_text, direction='fr_to_en')
    
    # Perform the translation
    try:
        pipe = translation_pipes['english']
        result = pipe(processed_text)
        english_text = result[0]['translation_text'].strip()
        
        # Postprocess the English text
        translated_text = postprocess_text(english_text, direction='fr_to_en')
        
        return translated_text
    except Exception as e:
        logger.error(f"Error during French to English translation: {e}")
        return {"Result": False, "Message": f"Translation error: {str(e)}"}

class MachineTranslation(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('translate_to', type=str, help='Language you want to translate to', required=True)
        parser.add_argument('english_text', type=str, help='English text for French translation', default='')
        parser.add_argument('french_text', type=str, help='French text for English translation', default='')
        parser.add_argument('ground_truth_english', type=str, help='Ground truth English text (optional)', default='')
        parser.add_argument('ground_truth_french', type=str, help='Ground truth French text (optional)', default='')
        
        args = parser.parse_args()
        translate_to = args.translate_to.lower()
        
        logger.info(f"Request received to translate to: {translate_to}")
        logger.info(f"English text: {args.english_text}")
        logger.info(f"French text: {args.french_text}")
        
        # Handle English to French translation
        if translate_to in ['french', 'fr']:
            if not args.english_text:
                logger.error("English text is missing for French translation")
                return {"Result": False, "Message": "English text is required"}, 400
            
            input_text = preprocess_text(args.english_text, direction='en_to_fr')
            ground_truth = preprocess_text(args.ground_truth_french, direction='en_to_fr')
            
            try:
                pipe = translation_pipes['french']
                result = pipe(input_text)
                french_text = result[0]['translation_text'].strip()
                french_text = postprocess_text(french_text, direction='en_to_fr')
                
                meteor_score_val, bleu_score_val = (None, None)
                if ground_truth:
                    meteor_score_val, bleu_score_val = metrics(french_text, ground_truth)
                
                logger.info(f"Translated to French: {french_text}")
                
                return jsonify({
                    "french_text": french_text,
                    "meteor_score": meteor_score_val,
                    "bleu_score": bleu_score_val
                })
            except Exception as e:
                logger.error(f"Error during English to French translation: {e}")
                return {"Result": False, "Message": f"Translation error: {str(e)}"}, 500
        
        # Handle French to English translation
        elif translate_to in ['english', 'en']:
            if not args.french_text:
                logger.error("French text is missing for English translation")
                return {"Result": False, "Message": "French text is required"}, 400
            
            try:
                english_text = translate_french_to_english(args.french_text)
                
                ground_truth = preprocess_text(args.ground_truth_english, direction='fr_to_en')
                meteor_score_val, bleu_score_val = (None, None)
                if ground_truth:
                    meteor_score_val, bleu_score_val = metrics(english_text, ground_truth)
                
                logger.info(f"Translated to English: {english_text}")
                
                return jsonify({
                    "english_text": english_text,
                    "meteor_score": meteor_score_val,
                    "bleu_score": bleu_score_val
                })
            except Exception as e:
                logger.error(f"Error during French to English translation: {e}")
                return {"Result": False, "Message": f"Translation error: {str(e)}"}, 500
        
        # Handle invalid language
        else:
            logger.error("Invalid translation language specified")
            return {"Result": False, "Message": "Invalid language specified"}, 400

#Add the resource to the Flask API
api.add_resource(MachineTranslation, '/translate')

if __name__ == '__main__':
     app.run(debug=True)

# # Example usage (can be removed or used for testing purposes)
# french_text = ""
# translated_text = translate_french_to_english(french_text)
# print(translated_text)
