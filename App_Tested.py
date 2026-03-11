"""
AI Interview System - Main Gradio Application
Complete with OCR resume parsing, adaptive difficulty, real-time rating, and comprehensive monitoring
Author: Kavya Bhardwaj
"""
import gradio as gr
import cv2
import numpy as np
from groq import Groq
from datetime import datetime
import json
from typing import Any
import speech_recognition as sr

# Import our custom modules
from interview_engine import EnhancedInterviewEngine
from monitoring import MonitoringSystem
from resume_parser import EnhancedResumeParser
from db import (
    init_db, seed_default_jobs_if_empty, list_jobs, get_job, upsert_job,
    create_interview, update_interview_transcript, complete_interview,
    list_completed_interviews_for_job
)


class InterviewState:
    """Global state management for the interview system"""
    def __init__(self):
        self.engine = None
        self.monitor = MonitoringSystem()
        self.parser = None
        self.interview_active = False
        self.input_mode = "both"  # "voice", "text", or "both"
        self.violations = {
            "tab_switches": 0,
            "copy_paste": 0,
            "looking_away": 0,
            "phone_detected": 0,
            "no_face": 0,
            "multiple_people": 0
        }
        self.start_time = None
        self.resume_data = None
        self.job_details = {}
        self.conversation_history = []
        self.active_interview_id = None
        self.candidate_profile = {"name": "", "email": ""}
        self.is_processing_audio = False  # Prevent duplicate submissions


state = InterviewState()


def parse_resume_file(file):
    """Parse uploaded resume file"""
    if file is None:
        return None, "Please upload a resume file"
    
    try:
        file_path = file.name
        state.parser = EnhancedResumeParser()
        resume_data = state.parser.parse_file(file_path)
        state.resume_data = resume_data
        
        if "error" in resume_data:
            return resume_data, f"Error: {resume_data['error']}"
        
        # Format resume data for display
        display_data = {
            "Name": resume_data.get("name", "Not found"),
            "Email": resume_data.get("email", "Not found"),
            "Phone": resume_data.get("phone", "Not found"),
            "Location": resume_data.get("location", "Not found"),
            "Skills Found": len(resume_data.get("skills", [])),
            "Top Skills": resume_data.get("skills", [])[:10],
            "Experience Entries": len(resume_data.get("experience", [])),
            "Education Entries": len(resume_data.get("education", []))
        }
        
        skills_by_category = resume_data.get("skills_by_category", {})
        if skills_by_category:
            display_data["Skills by Category"] = skills_by_category
        
        status = f"Resume parsed successfully! Found {len(resume_data.get('skills', []))} skills"
        return display_data, status
        
    except Exception as e:
        return None, f"Error parsing resume: {str(e)}"


def job_choices() -> list[tuple[str, int]]:
    jobs = list_jobs()
    return [(f"{j['title']} - {j['location']}", int(j['id'])) for j in jobs]


def refresh_jobs_dropdown(include_all_option: bool = False):
    choices = job_choices()
    if include_all_option:
        return gr.update(choices=[("All jobs", None)] + choices)
    return gr.update(choices=choices)


def start_interview(api_key, candidate_name, candidate_email, job_id, resume_file, input_mode):
    """Create interview and start (Apply → Interview)"""
    if not api_key:
        return ("Please enter your Groq API key",
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                [], [], None)
    
    if not job_id:
        return ("Please select a job",
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                [], [], None)
    
    if resume_file is None:
        return ("Please upload your resume",
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                [], [], None)
    
    try:
        job = get_job(int(job_id))
        if not job:
            return ("Selected job not found",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    [], [], None)
        
        # Parse resume 
        state.parser = EnhancedResumeParser()
        resume_data = state.parser.parse_file(resume_file.name)
        state.resume_data = resume_data
        
        if "error" in resume_data:
            return (f"Resume parse error: {resume_data['error']}",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    [], [], None)
        
        # Store job details
        state.job_details = {
            "position": job['title'],
            "location": job.get('location', ""),
            "skills": [s.strip() for s in job.get('required_skills', "").split(",") if s.strip()],
            "jd_text": job.get('jd_text', "")
        }
        
        state.candidate_profile = {
            "name": candidate_name or resume_data.get('name', ""),
            "email": candidate_email or resume_data.get('email', "")
        }
        
        # Update parser with API key and enrich categorization
        state.parser.groq_client = Groq(api_key=api_key)
        if state.resume_data.get('skills'):
            state.resume_data['skills_by_category'] = state.parser._categorize_skills_with_ai(
                state.resume_data['skills'],
                state.resume_data.get('raw_text', "")
            )
        
        # Initialize interview engine
        state.engine = EnhancedInterviewEngine(
            api_key=api_key,
            resume_data=state.resume_data,
            job_details=state.job_details
        )
        
        # Set input mode
        state.input_mode = (input_mode or "Both").lower()
        
        # Start interview
        state.interview_active = True
        state.start_time = datetime.now()
        
        # Get first question
        first_question = state.engine.get_next_question()
        
        # Add to conversation
        conversation = [{"role": "assistant", "content": first_question}]
        state.conversation_history = conversation
        
        # Persist interview
        state.active_interview_id = create_interview(
            candidate_name=state.candidate_profile.get('name', ""),
            candidate_email=state.candidate_profile.get('email', ""),
            job_id=int(job_id),
            input_mode=state.input_mode,
            resume_data=state.resume_data or {},
            transcript=conversation,
            violations=state.violations
        )
        
        return ("Interview started.",
                gr.update(visible=False),  # apply view
                gr.update(visible=True),   # interview view
                gr.update(visible=False),  # results view
                conversation,
                [],  # results JSON
                state.active_interview_id)
        
    except Exception as e:
        return (f"Error starting interview: {str(e)}",
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                [], [], None)


def process_video_frame(frame):
    """Process webcam frame for monitoring"""
    if frame is None or not state.interview_active:
        return frame, "Face: Not Active", "Eyes: Not Active", "Phone: Not Active", state.violations
    
    try:
        # Process frame through monitoring system
        processed_frame, results = state.monitor.process_frame(frame)
        
        # Update violations
        if not results['face_detected']:
            state.violations['no_face'] += 1
        if not results['eyes_detected'] and results['face_detected']:
            state.violations['looking_away'] += 1
        if results['phone_detected']:
            state.violations['phone_detected'] += 1
        
        face_status = f"Face Detected ✓ {1.0}" if results['face_detected'] else f"No Face ✗ {1.0}"
        eye_status = f"Eyes Detected ✓ {1.0}" if results['eyes_detected'] else f"Looking Away ✗ {1.0}"
        phone_status = f"No Phone ✓ {1.0}" if not results['phone_detected'] else f"Phone Detected ✗ {1.0}"
        
        return processed_frame, face_status, eye_status, phone_status, state.violations
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return frame, "Error ✗ {1.0}", "Error ✗ {1.0}", "Error ✗ {1.0}", state.violations


def process_text_answer(text_input, conversation):
    """Process text answer from candidate"""
    if not text_input or not state.interview_active:
        return conversation, "", None
    
    try:
        # Add user answer to conversation
        conversation.append({"role": "user", "content": text_input})
        
        # Get AI response
        ai_response = state.engine.get_next_question(text_input)
        
        # Add AI response
        conversation.append({"role": "assistant", "content": ai_response})
        
        state.conversation_history = conversation
        
        if state.active_interview_id:
            update_interview_transcript(state.active_interview_id, conversation, state.violations)
        
        return conversation, "", None
        
    except Exception as e:
        return conversation, "", f"Error: {str(e)}"


def process_audio_answer_auto(audio, conversation):
    """
    Automatically process voice answer when recording stops.
    Single-step: STT → LLM → response
    """
    # Prevent duplicate processing
    if state.is_processing_audio:
        return conversation, None, " Already processing..."
    
    if audio is None or not state.interview_active:
        return conversation, None, " No audio recorded or interview not active"
    
    # Lock processing
    state.is_processing_audio = True
    
    try:
        # Speech-to-text
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
        
        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            error_msg = " Could not understand audio. Please speak clearly or use text input."
            conversation.append({"role": "assistant", "content": error_msg})
            return conversation, None, error_msg
        except sr.RequestError as e:
            error_msg = f" Speech service error: {e}"
            conversation.append({"role": "assistant", "content": error_msg})
            return conversation, None, error_msg
        
        if not transcript.strip():
            error_msg = " Empty transcript. Please try again."
            conversation.append({"role": "assistant", "content": error_msg})
            return conversation, None, error_msg
        
        # Add user's voice answer to conversation
        conversation.append({"role": "user", "content": f"🎤 {transcript}"})
        
        # Get AI response
        ai_response = state.engine.get_next_question(transcript)
        conversation.append({"role": "assistant", "content": ai_response})
        
        # Update persistent storage
        state.conversation_history = conversation
        if state.active_interview_id:
            update_interview_transcript(
                state.active_interview_id,
                conversation,
                state.violations
            )
        
        # Success status
        success_status = f"Processed: \"{transcript[:60]}{'...' if len(transcript) > 60 else ''}\""
        
        return conversation, None, success_status
        
    except Exception as e:
        error_msg = f" Error: {str(e)}"
        return conversation, None, error_msg
    
    finally:
        # Unlock processing
        state.is_processing_audio = False


def ask_next_question(conversation):
    """Skip to next question"""
    if not state.interview_active:
        return conversation
    
    try:
        next_question = state.engine.get_next_question("I'd like to skip this question.")
        conversation.append({"role": "assistant", "content": next_question})
        
        state.conversation_history = conversation
        
        if state.active_interview_id:
            update_interview_transcript(state.active_interview_id, conversation, state.violations)
        
        return conversation
        
    except Exception as e:
        return conversation


def submit_code(code, conversation):
    """Submit and evaluate code"""
    if not code.strip() or not state.interview_active:
        return "Please write some code first.", conversation
    
    try:
        # Evaluate code with AI
        evaluation = state.engine.evaluate_code(code)
        
        conversation.append({
            "role": "user",
            "content": f"[CODE SUBMITTED]\n```python\n{code}\n```"
        })
        conversation.append({"role": "assistant", "content": evaluation})
        
        state.conversation_history = conversation
        
        if state.active_interview_id:
            update_interview_transcript(state.active_interview_id, conversation, state.violations)
        
        return evaluation, conversation
        
    except Exception as e:
        return f"Error: {str(e)}", conversation


def end_interview(conversation):
    """End interview → score → show results"""
    if not state.interview_active:
        return (gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True), {}, None)
    try:
        state.interview_active = False
        duration_seconds = None
        if state.start_time:
            duration = datetime.now() - state.start_time
            duration_seconds = int(duration.total_seconds())
        
        # Score full interview (single pass)
        result = state.engine.score_interview(
            transcript_messages=conversation or state.conversation_history,
            violations=state.violations,
            duration_seconds=duration_seconds
        )
        
        # Save report to file
        report_path = f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        if state.active_interview_id:
            update_interview_transcript(state.active_interview_id, conversation or [], state.violations)
            complete_interview(state.active_interview_id, result)
        
        return (gr.update(visible=False),  # apply view
                gr.update(visible=False),  # interview view
                gr.update(visible=True),   # results view
                result,
                report_path)
        
    except Exception as e:
        return (gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True), {"error": str(e)}, None)


def admin_login(pw: str):
    if (pw or "").strip() == "admin":
        return "Admin logged in.", gr.update(visible=True)
    return "Wrong password.", gr.update(visible=False)


def admin_save_job(job_id, title, location, skills, jd_text):
    jid = upsert_job(
        int(job_id) if job_id else None,
        title or "",
        location or "",
        skills or "",
        jd_text or ""
    )
    return f"Job saved (id={jid})", refresh_jobs_dropdown(), refresh_jobs_dropdown(include_all_option=True), gr.update(value=jid)


def log_tab_switch():
    state.violations['tab_switches'] += 1
    if state.active_interview_id:
        update_interview_transcript(state.active_interview_id, state.conversation_history, state.violations)
    return


def log_copy_event():
    state.violations['copy_paste'] += 1
    if state.active_interview_id:
        update_interview_transcript(state.active_interview_id, state.conversation_history, state.violations)
    return


def admin_load_results(job_id):
    jid = int(job_id) if job_id else None
    rows = list_completed_interviews_for_job(jid)
    
    ranked = []
    for r in rows:
        result = r.get('result') or {}
        ranked.append({
            "interview_id": r['id'],
            "job": r.get('job_title'),
            "candidate": r.get('candidate_name') or "",
            "email": r.get('candidate_email') or "",
            "overall": result.get('overall') if isinstance(result, dict) else None,
            "recommendation": result.get('recommendation') if isinstance(result, dict) else None,
            "ended_at": r.get('ended_at')
        })
    
    ranked.sort(key=lambda x: (x['overall'] is None, -(x['overall'] or 0)))
    return ranked


def create_interface():
    init_db()
    seed_default_jobs_if_empty()
    
    with gr.Blocks(
        title="AI Interview System",
        theme=gr.themes.Soft(),
        css="""
        #voice_recorder {
            border: 2px solid #10b981;
            border-radius: 8px;
            padding: 12px;
        }
        
        #voice_recorder button {
            font-size: 16px !important;
            padding: 12px 24px !important;
        }
        
        #audio_status {
            margin-top: 8px;
            padding: 8px;
            border-radius: 6px;
            background: #f3f4f6;
            min-height: 30px;
            text-align: center;
            font-weight: 500;
        }
        
        /* Recording state indicator */
        #voice_recorder[data-recording="true"] {
            border-color: #ef4444;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { border-color: #ef4444; }
            50% { border-color: #fca5a5; }
        }
        """
    ) as demo:
        gr.Markdown("# AI Interview System")
        gr.Markdown("**Candidate**: Apply → Interview → Results | **Admin**: Jobs + Rankings")
        
        interview_id_state = gr.State(None)
        
        with gr.Row():
            to_admin_btn = gr.Button(" Admin", variant="secondary")
            to_apply_btn = gr.Button(" Candidate", variant="secondary")
        
        # === APPLY VIEW ===
        with gr.Column(visible=True) as apply_view:
            gr.Markdown("## Apply for a job")
            api_key = gr.Textbox(label="Groq API Key", type="password", placeholder="Enter your Groq API key")
            
            with gr.Row():
                candidate_name = gr.Textbox(label="Name (optional)", placeholder="Your full name")
                candidate_email = gr.Textbox(label="Email (optional)", placeholder="you@example.com")
            
            job_dropdown = gr.Dropdown(label="Select Job", choices=job_choices(), value=None)
            resume_file = gr.File(label="Upload CV (PDF/DOCX/TXT/Image)", file_types=[".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"])
            input_mode = gr.Radio(label="Answer Input Mode (locked for interview)", choices=["Voice", "Text", "Both"], value="Both")
            apply_status = gr.Textbox(label="Status", interactive=False)
            start_btn = gr.Button(" Start Interview", variant="primary")
        
        # === INTERVIEW VIEW ===
        with gr.Column(visible=False) as interview_view:
            gr.Markdown("## Interview Room")
            
            with gr.Row():
                with gr.Column(scale=2):
                    webcam_feed = gr.Image(label="Monitoring (optional)", sources=["webcam"], streaming=True)
                    with gr.Row():
                        face_status = gr.Label(label="Face Detection")
                        eye_status = gr.Label(label="Eye Tracking")
                        phone_status = gr.Label(label="Phone Detection")
                    violations_display = gr.JSON(label="Violations Log")
                
                with gr.Column(scale=3):
                    conversation = gr.Chatbot(label="Interview Conversation", height=520)
                    
                    #  ONE-BUTTON VOICE SECTION 
                    with gr.Group():
                        gr.Markdown("### Voice Answer (One-Tap)")
                        
                        # Single audio component - auto-processes when recording stops
                        audio_input = gr.Audio(
                            label="Click to start/stop recording (auto-submits)",
                            sources=["microphone"],
                            type="filepath",
                            streaming=False,
                            # show_download_button=False,
                            container=True,
                            elem_id="voice_recorder"
                        )
                        
                        # Status indicator
                        audio_status = gr.Markdown("", elem_id="audio_status")
                    
                    # Text input (alternative)
                    with gr.Group():
                        gr.Markdown("### Text Answer")
                        text_input = gr.Textbox(
                            label="Or type your answer (text or code)",
                            placeholder="Type your answer here...",
                            lines=4
                        )
                        submit_text_btn = gr.Button("Submit Text", variant="primary")
                    
                    # Control buttons
                    with gr.Row():
                        next_question_btn = gr.Button(" Skip Question")
                        end_interview_btn = gr.Button(" End Interview", variant="stop")
        
        # === RESULTS VIEW ===
        with gr.Column(visible=False) as results_view:
            gr.Markdown("## Results")
            final_report = gr.JSON(label="Overall analysis")
            download_report = gr.File(label="Download Report (JSON)")
        
        # ADMIN VIEW ===
        with gr.Column(visible=False) as admin_view:
            gr.Markdown("## Admin Panel")
            admin_pw = gr.Textbox(label="Admin password", type="password", placeholder="(proto password is 'admin')")
            admin_login_btn = gr.Button("Login")
            admin_status = gr.Textbox(label="Status", interactive=False)
            
            admin_panel = gr.Column(visible=False)
            with admin_panel:
                gr.Markdown("### 💼 Jobs Management")
                with gr.Row():
                    job_id = gr.Number(label="Job ID (blank=new)", value=None, precision=0)
                    job_title = gr.Textbox(label="Title")
                with gr.Row():
                    job_location = gr.Textbox(label="Location")
                    job_skills = gr.Textbox(label="Required skills (comma separated)")
                job_jd = gr.Textbox(label="Job description (JD)", lines=6)
                save_job_btn = gr.Button(" Save Job", variant="primary")
                job_save_status = gr.Textbox(label="Job save status", interactive=False)
                
                gr.Markdown("### Results Ranking")
                admin_job_filter = gr.Dropdown(
                    label="Filter by job",
                    choices=[("All jobs", None)] + job_choices(),
                    value=None
                )
                refresh_results_btn = gr.Button(" Refresh results")
                results_table = gr.JSON(label="Ranked candidates")
        
        # Hidden buttons for client events (tab switch, copy/paste)
        tab_event_btn = gr.Button(visible=False, elem_id="tab_event_btn")
        copy_event_btn = gr.Button(visible=False, elem_id="copy_event_btn")
        
        # NAVIGATION 
        def show_admin():
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        
        def show_apply():
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        to_admin_btn.click(show_admin, outputs=[apply_view, interview_view, results_view, admin_view])
        to_apply_btn.click(show_apply, outputs=[apply_view, interview_view, results_view, admin_view])
        
        #  APPLY -> INTERVIEW
        start_btn.click(
            start_interview,
            inputs=[api_key, candidate_name, candidate_email, job_dropdown, resume_file, input_mode],
            outputs=[apply_status, apply_view, interview_view, results_view, conversation, final_report, interview_id_state]
        )
        
        # MONITORING 
        webcam_feed.stream(
            process_video_frame,
            inputs=[webcam_feed],
            outputs=[webcam_feed, face_status, eye_status, phone_status, violations_display]
        )
        
        # ONE-BUTTON VOICE: AUTO-PROCESS ON STOP
        audio_input.stop_recording(
            fn=process_audio_answer_auto,
            inputs=[audio_input, conversation],
            outputs=[conversation, audio_input, audio_status],
        )
        
        # TEXT SUBMISSION 
        submit_text_btn.click(
            process_text_answer,
            inputs=[text_input, conversation],
            outputs=[conversation, text_input, gr.State(None)]
        )
        
        next_question_btn.click(ask_next_question, inputs=[conversation], outputs=[conversation])
        
        # END -> RESULTS 
        end_interview_btn.click(
            end_interview,
            inputs=[conversation],
            outputs=[apply_view, interview_view, results_view, final_report, download_report]
        )
        
        #  ADMIN
        admin_login_btn.click(admin_login, inputs=[admin_pw], outputs=[admin_status, admin_panel])
        save_job_btn.click(
            admin_save_job,
            inputs=[job_id, job_title, job_location, job_skills, job_jd],
            outputs=[job_save_status, job_dropdown, admin_job_filter, job_id]
        )
        refresh_results_btn.click(admin_load_results, inputs=[admin_job_filter], outputs=[results_table])
        
        # CLIENT-SIDE EVENTS 
        tab_event_btn.click(log_tab_switch, outputs=[])
        copy_event_btn.click(log_copy_event, outputs=[])
        
        # Load client-side JavaScript for tab switching and copy/paste detection
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
            () => {
                const app = gradioApp();
                if (!app) return;
                
                const tabBtnContainer = app.querySelector("#tab_event_btn");
                const copyBtnContainer = app.querySelector("#copy_event_btn");
                const tabBtn = tabBtnContainer ? tabBtnContainer.querySelector("button") : null;
                const copyBtn = copyBtnContainer ? copyBtnContainer.querySelector("button") : null;
                
                if (tabBtn) {
                    document.addEventListener("visibilitychange", () => {
                        if (document.hidden) {
                            tabBtn.click();
                        }
                    });
                }
                
                const answerBox = app.querySelector("textarea[aria-label='Or type your answer (text or code)']");
                if (answerBox && copyBtn) {
                    ["copy", "paste"].forEach(ev => {
                        answerBox.addEventListener(ev, () => copyBtn.click());
                    });
                }
            }
            """
        )
        
        # text extraction and error handling
        conversation.change(
            fn=None,
            inputs=[conversation],
            outputs=[],
            js="""
            (messages) => {
                try {
                    if (!messages || !messages.length) return;
                    
                    const last = messages[messages.length - 1];
                    const role = last.role || (Array.isArray(last) ? last[0] : null);
                    let content = last.content || (Array.isArray(last) ? last[1] : null);
                    
                    // Only speak assistant messages
                    if (role !== "assistant" || !content) return;
                    
                    // Handle object content - extract text only
                    if (typeof content === 'object' && content !== null) {
                        // Try to extract meaningful text from object
                        if (content.text) content = content.text;
                        else if (content.message) content = content.message;
                        else return; // Skip if no readable text
                    }
                    
                    // Convert to string
                    content = String(content);
                    
                    // CLEAN UP: Remove JSON artifacts, markdown, special formatting
                    content = content
                        .replace(/```[\s\S]*?```/g, '') // Remove code blocks
                        .replace(/`[^`]+`/g, '') // Remove inline code
                        .replace(/\[Voice answer\]:/gi, '') // Remove voice labels
                        .replace(/🎤/g, '') // Remove microphone emoji
                        .replace(/\*\*/g, '') // Remove bold markdown
                        .replace(/\*/g, '') // Remove italic markdown
                        .replace(/#{1,6}\s/g, '') // Remove markdown headers
                        .replace(/\n{3,}/g, '\n\n') // Collapse multiple newlines
                        .replace(/\\n/g, ' ') // Replace escaped newlines
                        .replace(/\\"/g, '"') // Unescape quotes
                        .replace(/\\/g, '') // Remove remaining backslashes
                        .trim();
                    
                    // Skip if empty after cleaning
                    if (!content || content.length < 3) return;
                    
                    // Skip if content is just JSON or special chars
                    if (/^[\[\]{}",:]+$/.test(content)) return;
                    
                    // Initialize speech synthesis
                    if (!window.speechSynthesis) {
                        console.warn("Speech synthesis not supported");
                        return;
                    }
                    
                    const utterance = new SpeechSynthesisUtterance(content);
                    
                    // Configure voice settings
                    utterance.rate = 1.0;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    
                    // Set English voice
                    const voices = window.speechSynthesis.getVoices();
                    const englishVoice = voices.find(v => v.lang.startsWith('en'));
                    if (englishVoice) {
                        utterance.voice = englishVoice;
                    }
                    
                    // Cancel any ongoing speech
                    window.speechSynthesis.cancel();
                    
                    // Small delay to ensure cancel completes
                    setTimeout(() => {
                        window.speechSynthesis.speak(utterance);
                    }, 100);
                    
                    // Error handling
                    utterance.onerror = (event) => {
                        console.error('TTS error:', event);
                    };
                    
                } catch (e) {
                    console.warn('TTS failed:', e);
                }
            }
            """
        )

        
        return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
