from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from datetime import datetime
import json
import base64
from werkzeug.utils import secure_filename
import speech_recognition as sr
import requests
import tempfile
from pydub import AudioSegment
from pydub.utils import make_chunks
import threading
import time
import numpy as np
import librosa
import soundfile as sf

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": "https://ai-powered-public-speaking-game.vercel.app/"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)
# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'webm', 'ogg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('game_sessions', exist_ok=True)


def cleanup_session_data(session_id):
    """Enhanced cleanup function to remove all session data and audio files"""
    try:
        # Remove session JSON file
        session_file = os.path.join('game_sessions', f"session_{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
            print(f"Removed session file: {session_file}")
        
        # Remove all audio files for this session
        upload_dir = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                if filename.startswith(f"{session_id}_"):
                    file_path = os.path.join(upload_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"Removed audio file: {file_path}")
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")
        
        print(f"Session {session_id} cleanup completed")
        
    except Exception as e:
        print(f"Error during cleanup for session {session_id}: {e}")



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path):
    """Convert audio file to WAV format for speech recognition"""
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        
        # Convert to mono and set sample rate to 16kHz (good for speech recognition)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.export(temp_wav.name, format='wav')
        
        return temp_wav.name
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def analyze_vocal_energy(audio_path):
    """Analyze vocal energy levels from audio file"""
    try:
        # Load audio file with librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate RMS energy over time windows
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate statistics
        mean_energy = np.mean(rms)
        max_energy = np.max(rms)
        min_energy = np.min(rms)
        energy_std = np.std(rms)
        
        # Convert to decibel scale for better interpretation
        mean_energy_db = 20 * np.log10(mean_energy + 1e-10)
        max_energy_db = 20 * np.log10(max_energy + 1e-10)
        min_energy_db = 20 * np.log10(min_energy + 1e-10)
        
        # Normalize to 1-9 scale based on energy levels
        # These thresholds might need adjustment based on your microphone setup
        energy_level = normalize_energy_to_scale(mean_energy_db)
        
        # Calculate energy variation (how much the energy changes)
        energy_variation = energy_std / (mean_energy + 1e-10)
        
        # Detect energy segments for transition analysis
        time_frames = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        energy_segments = detect_energy_segments(rms, time_frames)
        
        return {
            'energy_level': int(energy_level),
            'mean_energy_db': float(mean_energy_db),
            'max_energy_db': float(max_energy_db),
            'min_energy_db': float(min_energy_db),
            'energy_variation': float(energy_variation),
            'energy_segments': energy_segments,
            'duration': float(len(y) / sr)
        }
        
    except Exception as e:
        print(f"Error analyzing vocal energy: {e}")
        return {
            'energy_level': 5,  # Default middle level
            'mean_energy_db': -20.0,
            'max_energy_db': -10.0,
            'min_energy_db': -30.0,
            'energy_variation': 0.1,
            'energy_segments': [],
            'duration': 0.0
        }

def normalize_energy_to_scale(energy_db):
    """Convert dB energy to 1-9 scale"""
    # These thresholds are based on typical speech levels
    # You may need to adjust these based on your setup
    if energy_db > -10:
        return 9  # Maximum energy
    elif energy_db > -15:
        return 8  # Very high
    elif energy_db > -20:
        return 7  # High
    elif energy_db > -25:
        return 6  # Energetic
    elif energy_db > -30:
        return 5  # Normal
    elif energy_db > -35:
        return 4  # Calm
    elif energy_db > -40:
        return 3  # Low
    elif energy_db > -45:
        return 2  # Very low
    else:
        return 1  # Whisper level

def detect_energy_segments(rms_values, time_frames):
    """Detect different energy segments in the audio"""
    segments = []
    
    if len(rms_values) < 3:
        return segments
    
    # Smooth the RMS values to reduce noise
    window_size = min(5, len(rms_values))
    smoothed_rms = np.convolve(rms_values, np.ones(window_size)/window_size, mode='same')
    
    current_level = normalize_energy_to_scale(20 * np.log10(smoothed_rms[0] + 1e-10))
    segment_start = time_frames[0]
    
    for i, (rms_val, time_val) in enumerate(zip(smoothed_rms[1:], time_frames[1:]), 1):
        energy_level = normalize_energy_to_scale(20 * np.log10(rms_val + 1e-10))
        
        # If energy level changes significantly, end current segment and start new one
        if abs(energy_level - current_level) >= 2:  # Threshold for significant change
            segments.append({
                'start_time': float(segment_start),
                'end_time': float(time_val),
                'energy_level': int(current_level),
                'duration': float(time_val - segment_start)
            })
            
            current_level = energy_level
            segment_start = time_val
    
    # Add the final segment
    if len(time_frames) > 0:
        segments.append({
            'start_time': float(segment_start),
            'end_time': float(time_frames[-1]),
            'energy_level': int(current_level),
            'duration': float(time_frames[-1] - segment_start)
        })
    
    return segments

def analyze_conductor_performance(session_data):
    """Analyze conductor game performance based on energy transitions"""
    prompts = session_data.get('prompts', [])
    game_settings = session_data.get('game_settings', {})
    
    if not prompts:
        return create_default_conductor_results()
    
    # Get energy changes from game settings
    energy_changes = game_settings.get('energy_changes', [])
    total_transitions = game_settings.get('total_transitions', 0)
    successful_transitions = game_settings.get('successful_transitions', 0)
    
    # Analyze each audio recording
    analyzed_prompts = []
    total_adaptation_time = 0
    successful_adaptations = 0
    energy_levels_used = set()
    consistency_scores = []
    
    for prompt_data in prompts:
        if prompt_data.get('energy_analysis'):
            analysis = prompt_data['energy_analysis']
            analyzed_prompts.append(analysis)
            
            # Track energy levels used
            energy_levels_used.add(analysis['energy_level'])
            
            # Calculate consistency (lower variation = more consistent)
            consistency_score = max(1, 10 - (analysis['energy_variation'] * 50))
            consistency_scores.append(min(10, consistency_score))
    
    # Calculate scores
    if total_transitions > 0:
        success_rate = (successful_transitions / total_transitions) * 100
        transition_score = min(10, (success_rate / 100) * 10)
    else:
        success_rate = 0
        transition_score = 5
    
    # Adaptability score based on successful transitions and response time
    avg_adaptation_speed = 2.0  # Default
    if energy_changes:
        adaptation_times = [change.get('transitionTime', 2000) for change in energy_changes if change.get('success')]
        if adaptation_times:
            avg_adaptation_speed = sum(adaptation_times) / len(adaptation_times) / 1000  # Convert to seconds
    
    adaptability_score = max(1, min(10, 10 - (avg_adaptation_speed - 1) * 2))
    
    # Consistency score
    avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 5
    
    # Energy range score (more levels used = better)
    energy_range_score = min(10, len(energy_levels_used) * 1.2)
    
    # Overall score
    overall_score = (transition_score + adaptability_score + avg_consistency + energy_range_score) / 4
    
    # Generate feedback
    feedback = generate_conductor_feedback(success_rate, adaptability_score, avg_consistency, energy_range_score, len(energy_levels_used))
    
    return {
        'game_type': 'conductor',
        'topic': prompts[0].get('prompt', 'Unknown topic') if prompts else 'Unknown topic',
        'duration': game_settings.get('duration', 180),
        'total_transitions': total_transitions,
        'successful_transitions': successful_transitions,
        'success_rate': round(success_rate, 1),
        'transition_score': round(transition_score, 1),
        'adaptability_score': round(adaptability_score, 1),
        'consistency_score': round(avg_consistency, 1),
        'energy_range_score': round(energy_range_score, 1),
        'avg_adaptation_speed': round(avg_adaptation_speed, 1),
        'energy_changes_detected': len(analyzed_prompts),
        'feedback': feedback,
        'overall_score': round(overall_score, 1)
    }

def generate_conductor_feedback(success_rate, adaptability, consistency, energy_range, levels_used):
    """Generate personalized feedback for conductor performance"""
    feedback_parts = []
    
    if success_rate >= 80:
        feedback_parts.append("Excellent energy control! You successfully adapted to most energy level changes.")
    elif success_rate >= 60:
        feedback_parts.append("Good energy adaptation with room for improvement in consistency.")
    else:
        feedback_parts.append("Focus on listening for energy cues and adapting your vocal intensity more dramatically.")
    
    if adaptability >= 8:
        feedback_parts.append("Your response time to energy changes is impressive.")
    elif adaptability >= 6:
        feedback_parts.append("Good adaptation speed, try to respond even more quickly to energy cues.")
    else:
        feedback_parts.append("Work on responding faster when you hear energy level changes.")
    
    if consistency >= 7:
        feedback_parts.append("Great vocal consistency when maintaining energy levels.")
    elif consistency >= 5:
        feedback_parts.append("Your energy control is developing - focus on maintaining steady levels when not transitioning.")
    else:
        feedback_parts.append("Practice maintaining consistent energy levels between transitions.")
    
    if levels_used >= 7:
        feedback_parts.append("Excellent use of the full vocal energy range!")
    elif levels_used >= 5:
        feedback_parts.append("Good energy range usage, try exploring even more extreme levels.")
    else:
        feedback_parts.append("Challenge yourself to use a wider range of vocal energy levels.")
    
    return " ".join(feedback_parts)

def create_default_conductor_results():
    """Create default results when no data is available"""
    return {
        'game_type': 'conductor',
        'topic': 'Unknown',
        'duration': 180,
        'total_transitions': 0,
        'successful_transitions': 0,
        'success_rate': 0,
        'transition_score': 5,
        'adaptability_score': 5,
        'consistency_score': 5,
        'energy_range_score': 5,
        'avg_adaptation_speed': 2.0,
        'energy_changes_detected': 0,
        'feedback': "Unable to analyze vocal energy patterns. Make sure your microphone is working properly.",
        'overall_score': 5
    }

def transcribe_audio(audio_path):
    """Transcribe audio to text using speech recognition"""
    recognizer = sr.Recognizer()
    
    try:
        # Convert to WAV if necessary
        wav_path = convert_to_wav(audio_path)
        if not wav_path:
            return None
            
        with sr.AudioFile(wav_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = recognizer.listen(source)
            
        try:
            # Use Google Speech Recognition (free tier)
            text = recognizer.recognize_google(audio)
            print(f"Transcribed: {text}")
            
            # Clean up temp file
            if wav_path != audio_path:
                os.unlink(wav_path)
                
            return text.lower().strip()
            
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return None
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def call_gemini(prompt):
    """Call Gemini API to analyze prompt and response"""
    try:
        if not GEMINI_API_KEY:
            print("Error: GEMINI_API_KEY environment variable not set")
            return None
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.9,
                "maxOutputTokens": 1000
            }
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract text from Gemini response structure
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text'].strip()
            
            print(f"Unexpected Gemini response structure: {result}")
            return None
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return None

def analyze_response_quality(prompt, user_response):
    """Analyze the quality of user's analogy response using Gemini (for Game 1)"""
    analysis_prompt = f"""
Analyze this analogy completion:

Prompt: "{prompt}"
User's response: "{user_response}"

Evaluate the response on a scale of 1-10 based on:
1. Creativity and originality
2. Relevance to the prompt
3. Logical connection/reasoning
4. Clarity and coherence

Provide your analysis in this exact JSON format:
{{
    "score": <number between 1-10>,
    "creativity": <number between 1-10>,
    "relevance": <number between 1-10>,
    "logic": <number between 1-10>,
    "clarity": <number between 1-10>,
    "feedback": "<brief explanation of the score>",
    "category": "<one word: excellent/good/fair/poor>"
}}

Only return the JSON, no other text.
"""
    
    llm_response = call_gemini(analysis_prompt)
    
    if llm_response:
        try:
            # Try to parse JSON from the response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = llm_response[start_idx:end_idx]
                analysis = json.loads(json_str)
                return analysis
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Raw response: {llm_response}")
            # Return default analysis
            return {
                "score": 5,
                "creativity": 5,
                "relevance": 5,
                "logic": 5,
                "clarity": 5,
                "feedback": "Unable to analyze response automatically",
                "category": "fair"
            }
    
    return {
        "score": 1,
        "creativity": 1,
        "relevance": 1,
        "logic": 1,
        "clarity": 1,
        "feedback": "Could not process response",
        "category": "poor"
    }

def process_audio_async(session_id, prompt_index, audio_path, prompt, game_type='analogy'):
    """Process audio file asynchronously"""
    try:
        print(f"Processing audio for session {session_id}, prompt {prompt_index}, game_type: {game_type}")
        
        session_file = os.path.join('game_sessions', f"session_{session_id}.json")
        
        if game_type == 'conductor':
            # For conductor game, analyze vocal energy
            energy_analysis = analyze_vocal_energy(audio_path)
            
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Find and update the corresponding prompt
                for prompt_data in session_data.get('prompts', []):
                    if prompt_data.get('prompt_index') == prompt_index:
                        prompt_data['energy_analysis'] = energy_analysis
                        prompt_data['processed_at'] = datetime.utcnow().isoformat()
                        print(f"Energy analysis for prompt {prompt_index}: Level {energy_analysis['energy_level']}")
                        break
                
                session_data['last_updated'] = datetime.utcnow().isoformat()
                
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
        
        else:
            # For analogy game, transcribe and analyze content
            transcription = transcribe_audio(audio_path)
            
            if transcription:
                # Analyze response quality
                analysis = analyze_response_quality(prompt, transcription)
                
                # Update session file with transcription and analysis
                if os.path.exists(session_file):
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    # Find and update the corresponding prompt
                    for prompt_data in session_data.get('prompts', []):
                        if prompt_data.get('prompt_index') == prompt_index:
                            prompt_data['transcription'] = transcription
                            prompt_data['analysis'] = analysis
                            prompt_data['processed_at'] = datetime.utcnow().isoformat()
                            break
                    
                    session_data['last_updated'] = datetime.utcnow().isoformat()
                    
                    with open(session_file, 'w') as f:
                        json.dump(session_data, f, indent=2)
                    
                    print(f"Processed prompt {prompt_index}: '{transcription}' -> Score: {analysis['score']}")
        
    except Exception as e:
        print(f"Error processing audio: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check Gemini connection
    gemini_status = "disconnected"
    try:
        if GEMINI_API_KEY:
            # Test API with a simple request
            test_response = call_gemini("Hello")
            if test_response:
                gemini_status = "connected"
        else:
            gemini_status = "no_api_key"
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'gemini_status': gemini_status
    })

@app.route('/api/voice/upload', methods=['POST'])
def upload_voice():
    """Upload voice recording with prompt data"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        required_fields = ['session_id', 'prompt', 'prompt_index', 'audio_data', 'audio_format']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        session_id = data['session_id']
        prompt = data['prompt']
        prompt_index = data['prompt_index']
        audio_data = data['audio_data']
        audio_format = data['audio_format']
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        response_time = data.get('response_time', 0)
        game_settings = data.get('game_settings', {})
        
        # Determine game type from settings
        game_type = game_settings.get('game_type', 'analogy')
        
        if audio_format not in ALLOWED_EXTENSIONS:
            return jsonify({
                'error': 'Invalid audio format',
                'allowed_formats': list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Create unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{session_id}_prompt_{prompt_index}_{unique_id}.{audio_format}"
        secure_name = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
        
        # Decode and save audio file
        try:
            audio_bytes = base64.b64decode(audio_data)
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
        except Exception as e:
            return jsonify({
                'error': 'Failed to decode audio data',
                'details': str(e)
            }), 400
        
        # Save session data
        sessions_dir = 'game_sessions'
        session_file = os.path.join(sessions_dir, f"session_{session_id}.json")
        
        prompt_data = {
            'prompt': prompt,
            'prompt_index': prompt_index,
            'response_time': response_time,
            'timestamp': timestamp,
            'filename': secure_name,
            'file_size': len(audio_bytes),
            'upload_time': datetime.utcnow().isoformat(),
            'processed': False,
            'game_type': game_type
        }
        
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                existing_data = json.load(f)
            
            if 'prompts' not in existing_data:
                existing_data['prompts'] = []
            existing_data['prompts'].append(prompt_data)
            existing_data['last_updated'] = datetime.utcnow().isoformat()
            
            with open(session_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        else:
            new_session = {
                'session_id': session_id,
                'created_at': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat(),
                'game_settings': game_settings,
                'game_type': game_type,
                'prompts': [prompt_data],
                'status': 'active'
            }
            
            with open(session_file, 'w') as f:
                json.dump(new_session, f, indent=2)
        
        # Start async processing
        threading.Thread(
            target=process_audio_async,
            args=(session_id, prompt_index, filepath, prompt, game_type),
            daemon=True
        ).start()
        
        return jsonify({
            'status': 'success',
            'message': 'Voice recording uploaded successfully',
            'data': {
                'session_id': session_id,
                'prompt': prompt,
                'prompt_index': prompt_index,
                'filename': secure_name,
                'game_type': game_type,
                'processing': True
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e)
        }), 500

@app.route('/api/session/<session_id>/complete', methods=['POST'])
def complete_session(session_id):
    """Mark session as complete and return processed results"""
    try:
        data = request.get_json() or {}
        
        session_file = os.path.join('game_sessions', f"session_{session_id}.json")
        
        if not os.path.exists(session_file):
            return jsonify({'error': 'Session not found'}), 404
        
        # Wait for processing to complete (with timeout)
        max_wait = 30  # 30 seconds timeout
        wait_time = 0
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        game_type = session_data.get('game_type', 'analogy')
        
        if game_type == 'conductor':
            # For conductor game, wait for energy analysis to complete
            while wait_time < max_wait:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                prompts = session_data.get('prompts', [])
                unprocessed = [p for p in prompts if not p.get('energy_analysis')]
                
                if not unprocessed:
                    break
                    
                time.sleep(1)
                wait_time += 1
            
            # Analyze conductor performance
            final_results = analyze_conductor_performance(session_data)
            
        else:
            # For analogy game, wait for transcription to complete
            while wait_time < max_wait:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                prompts = session_data.get('prompts', [])
                unprocessed = [p for p in prompts if not p.get('transcription')]
                
                if not unprocessed:
                    break
                    
                time.sleep(1)
                wait_time += 1
            
            # Calculate analogy game results
            final_results = calculate_analogy_results(session_data)
        
        # Update session with completion data
        session_data['completed_at'] = datetime.utcnow().isoformat()
        session_data['final_results'] = final_results
        session_data['status'] = 'completed'
        session_data['last_updated'] = datetime.utcnow().isoformat()
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        cleanup_session_data(session_id)
        return jsonify({
            'status': 'success',
            'message': 'Session completed successfully',
            'data': final_results
        }), 200
        
    except Exception as e:
        cleanup_session_data(session_id)
        return jsonify({
            'status': 'error',
            'message': 'Failed to complete session',
            'error': str(e)
        }), 500

def calculate_analogy_results(session_data):
    """Calculate results for analogy game (Game 1)"""
    prompts = session_data.get('prompts', [])
    total_prompts = len(prompts)
    
    # Count successful responses (those with transcriptions)
    successful_responses = [p for p in prompts if p.get('transcription')]
    response_count = len(successful_responses)
    
    # Calculate average scores
    scored_responses = [p for p in prompts if p.get('analysis', {}).get('score')]
    avg_score = sum(p['analysis']['score'] for p in scored_responses) / len(scored_responses) if scored_responses else 0
    
    # Calculate response rate
    response_rate = (response_count / total_prompts * 100) if total_prompts > 0 else 0
    
    # Calculate average response time
    timed_responses = [p for p in prompts if p.get('response_time')]
    avg_response_time = sum(p['response_time'] for p in timed_responses) / len(timed_responses) if timed_responses else 0
    
    # Categorize responses
    categories = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
    for prompt in scored_responses:
        category = prompt['analysis'].get('category', 'fair')
        if category in categories:
            categories[category] += 1
    
    # Get top responses
    top_responses = sorted(
        [p for p in scored_responses if p.get('transcription')],
        key=lambda x: x['analysis']['score'],
        reverse=True
    )[:3]
    
    return {
        'total_prompts': total_prompts,
        'completed_prompts': response_count,
        'response_rate': round(response_rate, 1),
        'avg_response_time': round(avg_response_time),
        'avg_quality_score': round(avg_score, 1),
        'score_breakdown': {
            'creativity': round(sum(p['analysis']['creativity'] for p in scored_responses) / len(scored_responses) if scored_responses else 0, 1),
            'relevance': round(sum(p['analysis']['relevance'] for p in scored_responses) / len(scored_responses) if scored_responses else 0, 1),
            'logic': round(sum(p['analysis']['logic'] for p in scored_responses) / len(scored_responses) if scored_responses else 0, 1),
            'clarity': round(sum(p['analysis']['clarity'] for p in scored_responses) / len(scored_responses) if scored_responses else 0, 1)
        },
        'category_breakdown': categories,
        'top_responses': [
            {
                'prompt': resp['prompt'],
                'response': resp['transcription'],
                'score': resp['analysis']['score'],
                'feedback': resp['analysis']['feedback']
            }
            for resp in top_responses
        ],
        'missed_prompts': [
            p['prompt'] for p in prompts if not p.get('transcription')
        ]
    }

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session data by session ID"""
    try:
        session_file = os.path.join('game_sessions', f"session_{session_id}.json")
        
        if not os.path.exists(session_file):
            return jsonify({'error': 'Session not found'}), 404
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        return jsonify({
            'status': 'success',
            'data': session_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve session',
            'error': str(e)
        }), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all game sessions"""
    try:
        sessions_dir = 'game_sessions'
        if not os.path.exists(sessions_dir):
            return jsonify({'status': 'success', 'data': []}), 200
        
        sessions = []
        for filename in os.listdir(sessions_dir):
            if filename.endswith('.json') and filename.startswith('session_'):
                filepath = os.path.join(sessions_dir, filename)
                with open(filepath, 'r') as f:
                    session_data = json.load(f)
                    sessions.append({
                        'session_id': session_data['session_id'],
                        'created_at': session_data['created_at'],
                        'last_updated': session_data['last_updated'],
                        'status': session_data.get('status', 'unknown'),
                        'game_type': session_data.get('game_type', 'unknown'),
                        'total_prompts': len(session_data.get('prompts', [])),
                        'processed_prompts': len([p for p in session_data.get('prompts', []) if p.get('transcription') or p.get('energy_analysis')]),
                        'game_settings': session_data.get('game_settings', {})
                    })
        
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'data': sessions,
            'total': len(sessions)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Failed to list sessions',
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large', 'max_size': '16MB'}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'message': 'Something went wrong on our end'}), 500

if __name__ == '__main__':
    print("Starting Enhanced Voice Processing API with Gemini...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Sessions folder: {os.path.abspath('game_sessions')}")
    print(f"Gemini API URL: {GEMINI_API_URL}")
    
    # Check if Gemini API key is set
    if GEMINI_API_KEY:
        print("✓ Gemini API key found")
        
        # Test Gemini connection
        try:
            test_response = call_gemini("Hello, this is a test.")
            if test_response:
                print("✓ Gemini API connection successful")
            else:
                print("✗ Gemini API connection failed - check your API key")
        except Exception as e:
            print(f"✗ Gemini API test failed: {e}")
    else:
        print("✗ Gemini API key not found - set GEMINI_API_KEY environment variable")
        print("  Example: export GEMINI_API_KEY='your_api_key_here'")
    
    # Check if required audio processing libraries are available
    try:
        import librosa
        import soundfile as sf
        print("✓ Audio processing libraries (librosa, soundfile) available")
    except ImportError as e:
        print(f"✗ Audio processing libraries missing: {e}")
        print("Install with: pip install librosa soundfile")
    
    print("\nAvailable endpoints:")
    print("  GET  /health - Health check")
    print("  POST /api/voice/upload - Upload voice recording")
    print("  GET  /api/session/<session_id> - Get session data")
    print("  GET  /api/sessions - List all sessions")
    print("  POST /api/session/<session_id>/complete - Complete session with AI analysis")
    print("\nGame types supported:")
    print("  - analogy: Content-based analysis for Game 1")
    print("  - conductor: Energy-based analysis for Game 2")
    print("\nNote: Make sure to set your GEMINI_API_KEY environment variable!")
    
    # ✅ Use Render's PORT if available, otherwise default to 5000
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port, debug=False)
