from dotenv import load_dotenv
load_dotenv()

#VoiceBot UI with Gradio
import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query, analyze_text_only, get_medication_advice
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

load_dotenv()

system_prompt="""You are role-playing as a professional doctor for educational purposes only. Analyze the provided image or query for potential medical issues. Respond as if speaking directly to a patient: be empathetic, concise (1-3 sentences), and professional. Avoid AI-like language, numbers, special characters, and repetitive phrases—vary your wording to sound natural and conversational, without fixed openings like "With what I see." If identifying a condition, suggest general, evidence-based remedies and always recommend consulting a real healthcare professional. Start your response immediately without preamble."""


def process_inputs(audio_filepath, image_filepath, language, elevenlabs_voice):
    speech_to_text_output = ""
    if audio_filepath:
        speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                                                     audio_filepath=audio_filepath,
                                                     stt_model="whisper-large-v3",
                                                     language=language)

    # Handle the inputs
    if image_filepath and audio_filepath:
        doctor_response = analyze_image_with_query(query=system_prompt + speech_to_text_output, encoded_image=encode_image(image_filepath), model="meta-llama/llama-4-scout-17b-16e-instruct")
    elif image_filepath:
        doctor_response = analyze_image_with_query(query=system_prompt, encoded_image=encode_image(image_filepath), model="meta-llama/llama-4-scout-17b-16e-instruct")
    elif audio_filepath:
        doctor_response = analyze_text_only(query=system_prompt + speech_to_text_output, model="meta-llama/llama-4-scout-17b-16e-instruct")
    else:
        doctor_response = "No audio or image provided for analysis."

    medication_advice = get_medication_advice(doctor_response)

    voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath="final.mp3", voice=elevenlabs_voice)

    return speech_to_text_output, doctor_response, medication_advice, voice_of_doctor


def get_tips():
    tips_query = "Provide 3 quick general health tips for patients in a concise paragraph."
    tips = analyze_text_only(query=tips_query, model="meta-llama/llama-4-scout-17b-16e-instruct")
    return tips


# Create the interface with advanced medical UI
with gr.Blocks(title="MedEd", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"), css=".gradio-container { font-family: 'Gilroy', sans-serif; }") as iface:
    gr.Markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="font-size: 3em; margin: 0; font-weight: bold;"> MedEd </h1>
        <h2 style="font-size: 1.5em; margin: 10px 0; font-weight: 300;">Medical Education AI Assistant</h2>
        <p style="font-size: 1.2em; margin: 20px 0; line-height: 1.6;">Empowering healthcare education through AI-driven diagnostics and personalized medication guidance.</p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
            <div style="text-align: center;">
                <h3 style="margin: 0; font-size: 1.1em;">Syed Mohib Ur Rehman</h3>
                <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Team Lead</p>
            </div>
            <div style="text-align: center;">
                <h3 style="margin: 0; font-size: 1.1em;">Aun Rashid</h3>
                <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Developer</p>
            </div>
            <div style="text-align: center;">
                <h3 style="margin: 0; font-size: 1.1em;">Ayaan Hussain</h3>
                <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Analyst</p>
            </div>
            <div style="text-align: center;">
                <h3 style="margin: 0; font-size: 1.1em;">Bilal Khan</h3>
                <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Researcher</p>
            </div>
        </div>
    </div>
    """)
    gr.Markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3 style="color: #4a5568; font-size: 1.3em;">Get Professional Medical Insights</h3>
        <p style="color: #718096; font-size: 1.1em;">Upload an image of a medical condition or record your voice to receive AI-powered analysis, diagnosis, and medication recommendations.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            language = gr.Dropdown(choices=["en", "es", "ur"], value="en", label="🌐 Language")
            elevenlabs_voice = gr.Dropdown(choices=["Aria", "Josh", "Domi"], value="Aria", label="🎭 ElevenLabs Voice")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Voice Input")
            image_input = gr.Image(type="filepath", label="📷 Image Input")

        with gr.Column(scale=2):
            speech_to_text = gr.Textbox(label="📝 Transcribed Speech", lines=2, interactive=False)
            doctor_response = gr.Textbox(label="👨‍⚕️ Doctor's Diagnosis & Advice", lines=4, interactive=False)
            medication_advice = gr.Textbox(label="💊 Medication Advice", lines=3, interactive=False)
            audio_output = gr.Audio(value=None, autoplay=True, label="🔊 Doctor's Voice Response")
            tips_textbox = gr.Textbox(label="💡 Quick Medical Tips", lines=3, interactive=False)

    submit_btn = gr.Button("🔍 Analyze", variant="primary")
    tips_btn = gr.Button("💡 Get Quick Tips")

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input, language, elevenlabs_voice],
        outputs=[speech_to_text, doctor_response, medication_advice, audio_output]
    )

    tips_btn.click(
        fn=get_tips,
        inputs=[],
        outputs=[tips_textbox]
    )

iface.launch(debug=True)
