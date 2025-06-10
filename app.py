import requests
import streamlit as st
from configs.load_tools_config import LoadToolsConfig
from src.voice.speech_io import transcribe_audio, synthesize_speech
from streamlit_chat_widget import chat_input_widget

# Load configuration
tool_cfg = LoadToolsConfig()
API_URL = "http://127.0.0.1:8000/chat/" # local development API URL
#API_URL = "http://0.0.0.0:8000/chat/" # docker image api key

# Set up the page
st.set_page_config(page_title="Medical AI Chatbot 🧬", page_icon="👨‍⚕️")
st.title("👨‍⚕️ Medical AI Assistant ⚕️🧬")
st.markdown("Ask your question by typing or clicking the mic icon 🎤.")

st.sidebar.header("⚙️ Configuration")
selected_model = st.sidebar.selectbox("Select LLM Model", tool_cfg.llm_models)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["assistant"])

st.markdown("---")  # use horizontal line instead of bottom() to keep layout stable
try:
    user_input = chat_input_widget()
except Exception as e:
    st.error(f"⚠️ Chat widget error: {e}")
    user_input = None

# Process the user's input from the widget
if user_input:
    user_question = ""

    if "text" in user_input:
        user_question = user_input["text"]
    elif "audioFile" in user_input:
        audio_bytes = bytes(user_input["audioFile"])
        st.audio(audio_bytes, format="audio/mp3")
        with st.spinner("🔊 Transcribing audio..."):
            try:
                user_question = transcribe_audio(audio_bytes)
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
                user_question = ""

    # Display and process question
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("👨‍⚕️ Thinking..."):
            try:
                response = requests.post(API_URL, json={"question": user_question, "model_name": selected_model})
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "")

                    # Format if response includes agent
                    if answer.startswith("Agent:"):
                        agent_line, _, actual_answer = answer.partition("Answer:")
                        agent_name = agent_line.replace("Agent:", "").strip()
                        actual_answer = actual_answer.strip()
                        formatted_answer = f"👨‍⚕️ **Agent:** {agent_name}\n\n💬 **Answer:**\n{actual_answer}"
                        text_to_speak = actual_answer
                    else:
                        formatted_answer = answer
                        text_to_speak = answer

                    with st.chat_message("assistant"):
                        st.markdown(formatted_answer)
                        with st.spinner("🔊 Generating audio..."):
                            try:
                                audio_response = synthesize_speech(text_to_speak)
                                if audio_response:
                                    st.audio(audio_response, format="audio/mp3")
                                else:
                                    st.warning("⚠️ Failed to generate audio.")
                            except Exception as e:
                                st.warning(f"⚠️ TTS error: {e}")

                    # Save chat to session
                    st.session_state.chat_history.append({
                        "user": user_question,
                        "assistant": formatted_answer
                    })
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"❌ Backend request failed: {e}")