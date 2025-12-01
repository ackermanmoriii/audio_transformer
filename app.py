import io
import math
import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, send_file
from pydub import AudioSegment
import noisereduce as nr

# --- The Professional Audio Library ---
from pedalboard import (
    Pedalboard, 
    Compressor, 
    Limiter, 
    HighpassFilter, 
    LowpassFilter, 
    HighShelfFilter, 
    LowShelfFilter,
    Distortion,
    NoiseGate
)

app = Flask(__name__)

# --- AUDIO UTILITIES ---

def convert_audio_to_numpy(audio_segment: AudioSegment):
    """
    Converts Pydub AudioSegment to a float32 numpy array (channels, samples)
    Required for Pedalboard and Noisereduce.
    """
    channel_sounds = audio_segment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    
    return fp_arr.T, audio_segment.frame_rate

def convert_numpy_to_audio(data: np.ndarray, sample_rate: int):
    """
    Converts float32 numpy array back to Pydub AudioSegment.
    """
    data = np.clip(data, -1.0, 1.0)
    
    pcm_data = (data * 32767).astype(np.int16)
    
    if pcm_data.shape[0] == 2: # Stereo
        pcm_data = pcm_data.T.flatten()
        channels = 2
    else: # Mono
        pcm_data = pcm_data.flatten()
        channels = 1
        
    return AudioSegment(
        data=pcm_data.tobytes(), 
        sample_width=2, 
        frame_rate=sample_rate, 
        channels=channels
    )

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio_file' not in request.files:
        return "No file uploaded", 400
    
    voice_file = request.files['audio_file']
    if voice_file.filename == '':
        return "No selected file", 400

    try:
        # ==========================================
        # STAGE 1: VOICE PROCESSING (Pedalboard/DSP)
        # ==========================================
        
        # 1. Load Voice
        original_voice = AudioSegment.from_file(voice_file)
        
        # 2. Convert to Numpy
        audio_data, sample_rate = convert_audio_to_numpy(original_voice)
        
        # 3. Noise Reduction (Spectral Gating)
        if request.form.get('enable_nr') == 'true':
            intensity = float(request.form.get('val_nr', 0)) / 100.0
            if intensity > 0:
                audio_data = nr.reduce_noise(
                    y=audio_data, 
                    sr=sample_rate, 
                    prop_decrease=intensity,
                    stationary=True
                )

        # 4. Pedalboard Rack
        board_modules = []

        # A. Vintage
        if request.form.get('enable_vintage') == 'true':
            v_val = float(request.form.get('val_vintage', 0)) / 100.0
            board_modules.append(HighpassFilter(cutoff_frequency_hz=300 + (200 * v_val)))
            board_modules.append(LowpassFilter(cutoff_frequency_hz=3500 - (2000 * v_val)))
            drive_db = 20 * v_val
            board_modules.append(Distortion(drive_db=drive_db))

        # B. Mic Sim
        if request.form.get('enable_mic') == 'true':
            board_modules.append(HighpassFilter(cutoff_frequency_hz=80))
            board_modules.append(HighShelfFilter(cutoff_frequency_hz=4000, gain_db=3))
            board_modules.append(LowShelfFilter(cutoff_frequency_hz=200, gain_db=2))

        # C. Podcast Voice
        if request.form.get('enable_podcast') == 'true':
            p_val = float(request.form.get('val_podcast', 0)) / 100.0
            board_modules.append(NoiseGate(threshold_db=-40, ratio=4, release_ms=200))
            ratio = 2 + (4 * p_val) 
            board_modules.append(Compressor(threshold_db=-16, ratio=ratio, attack_ms=1, release_ms=100))
            board_modules.append(Limiter(threshold_db=-1.0))

        # Apply Effects
        if board_modules:
            board = Pedalboard(board_modules)
            processed_data = board(audio_data, sample_rate)
        else:
            processed_data = audio_data

        # Convert back to Pydub for Mixing
        processed_voice_seg = convert_numpy_to_audio(processed_data, sample_rate)

        # ==========================================
        # STAGE 2: BACKGROUND MUSIC MIXING
        # ==========================================
        
        bg_file = request.files.get('bg_file')
        
        if bg_file and bg_file.filename != '':
            bg_file.seek(0)
            music_seg = AudioSegment.from_file(bg_file)
            
            # Match Frame Rate
            if music_seg.frame_rate != processed_voice_seg.frame_rate:
                music_seg = music_seg.set_frame_rate(processed_voice_seg.frame_rate)
            
            # --- 2A. TRIM & LOOP ---
            try:
                start_sec = float(request.form.get('bg_start', 0))
                end_sec = float(request.form.get('bg_end', 0))
            except (ValueError, TypeError):
                start_sec = 0
                end_sec = 0

            # Cut Selection
            if end_sec > 0 and end_sec > start_sec:
                music_seg = music_seg[start_sec*1000 : end_sec*1000]
            else:
                music_seg = music_seg[start_sec*1000:]
            
            # Loop to cover voice
            if len(music_seg) < len(processed_voice_seg):
                loops = int(len(processed_voice_seg) / len(music_seg)) + 1
                music_seg = music_seg * loops
            
            # Initial Trim to Voice Length
            music_seg = music_seg[:len(processed_voice_seg)]
            
            # --- 2B. VOLUME ---
            bg_vol_percent = float(request.form.get('bg_vol', 20))
            if bg_vol_percent <= 0:
                music_seg = music_seg - 100
            else:
                gain_db = 20 * math.log10(bg_vol_percent / 100.0)
                music_seg = music_seg + gain_db

            # --- 2C. FADE OUT LOGIC (New Feature) ---
            try:
                # When to start fading (in seconds)
                fade_start_sec = float(request.form.get('bg_fade_start', 0))
                # How long the slope is (in seconds)
                fade_len_sec = float(request.form.get('bg_fade_len', 3)) 
            except (ValueError, TypeError):
                fade_start_sec = 0
                fade_len_sec = 3

            # Apply Fade if user requested it
            if fade_start_sec > 0:
                fade_start_ms = fade_start_sec * 1000
                fade_len_ms = fade_len_sec * 1000
                total_duration_ms = fade_start_ms + fade_len_ms

                # If the fade end point is before the voice ends, we cut the music there
                if total_duration_ms < len(music_seg):
                    music_seg = music_seg[:int(total_duration_ms)]
                    music_seg = music_seg.fade_out(int(fade_len_ms))
                else:
                    # If fade starts near the end, just fade out the end of the existing clip
                    music_seg = music_seg.fade_out(int(fade_len_ms))

            # --- 2D. OVERLAY ---
            final_output = processed_voice_seg.overlay(music_seg, position=0)
        else:
            final_output = processed_voice_seg

        # ==========================================
        # STAGE 3: EXPORT
        # ==========================================
        buffer = io.BytesIO()
        final_output.export(buffer, format="mp3", bitrate="192k")
        buffer.seek(0)
        
        return send_file(
            buffer, 
            mimetype="audio/mpeg", 
            as_attachment=False, 
            download_name="final_mix.mp3"
        )

    except Exception as e:
        print(f"Server Error: {e}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)