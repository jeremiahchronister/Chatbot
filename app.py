import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
import base64
from pathlib import Path
import mimetypes

# Page configuration
st.set_page_config(page_title="Multi-AI Chatbot", page_icon="🤖", layout="wide")

# Define agent personas with starter prompts
AGENTS = {
    "General Assistant": {
        "icon": "🤖",
        "system_prompt": "You are a helpful, friendly, and knowledgeable AI assistant.",
        "description": "General purpose assistant for any task",
        "starters": [
            "Help me brainstorm ideas for a project",
            "Explain a complex topic in simple terms",
            "Summarize the key points from this document"
        ]
    },
    "Resume Helper": {
        "icon": "📝",
        "system_prompt": "You are an expert career coach and resume writer with 15+ years of experience. You help people create compelling resumes, optimize them for ATS systems, craft impactful bullet points using the STAR method, and provide tailored advice for different industries and career levels.",
        "description": "Expert help with resumes, cover letters, and career advice",
        "starters": [
            "Review my resume and suggest improvements",
            "Help me write compelling bullet points for my experience",
            "Create a cover letter for this job description"
        ]
    },
    "Finance Advisor": {
        "icon": "💰",
        "system_prompt": "You are a knowledgeable financial advisor specializing in personal finance, investing, budgeting, and financial planning. You provide clear, actionable advice on saving, investing, retirement planning, tax strategies, and wealth management. You always remind users to consult with licensed professionals for personalized advice.",
        "description": "Personal finance, investing, and budgeting guidance",
        "starters": [
            "Help me create a monthly budget",
            "Explain different investment strategies for beginners",
            "What should I consider for retirement planning?"
        ]
    },
    "Healthcare Assistant": {
        "icon": "🏥",
        "system_prompt": "You are a healthcare information assistant with expertise in general health, wellness, nutrition, and medical information. You provide evidence-based health information, explain medical concepts clearly, and offer wellness tips. You always emphasize that your information is educational and users should consult healthcare professionals for medical advice, diagnosis, or treatment.",
        "description": "Health information, wellness tips, and medical education",
        "starters": [
            "What are some evidence-based ways to improve sleep?",
            "Explain this medical term or condition",
            "Suggest a balanced meal plan for better nutrition"
        ]
    },
    "Coding Mentor": {
        "icon": "💻",
        "system_prompt": "You are an experienced software engineer and coding mentor. You help with programming questions, debug code, explain concepts clearly, provide best practices, and guide learners through technical challenges. You write clean, well-documented code and explain your reasoning.",
        "description": "Programming help, code review, and technical guidance",
        "starters": [
            "Review my code and suggest improvements",
            "Help me debug this error",
            "Explain this programming concept with examples"
        ]
    },
    "Creative Writer": {
        "icon": "✍️",
        "system_prompt": "You are a creative writing coach and accomplished author. You help with storytelling, creative writing, content creation, editing, and developing engaging narratives. You provide constructive feedback, suggest improvements, and inspire creativity.",
        "description": "Creative writing, storytelling, and content creation",
        "starters": [
            "Help me develop a compelling story idea",
            "Edit and improve this piece of writing",
            "Give me creative writing prompts to practice"
        ]
    }
}

# Helper function to encode images
def encode_image(image_file):
    """Encode image to base64"""
    return base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to read text documents
def read_document(doc_file):
    """Read text from document files"""
    file_extension = Path(doc_file.name).suffix.lower()
    
    try:
        if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml']:
            return doc_file.read().decode('utf-8')
        elif file_extension == '.pdf':
            # For PDF, we'll return a message that it's attached
            return f"[PDF Document: {doc_file.name}]"
        else:
            return f"[Document: {doc_file.name}]"
    except Exception as e:
        return f"[Could not read {doc_file.name}: {str(e)}]"

# Title
st.title("🤖 Multi-AI Chatbot with Personas")
st.markdown("Choose an AI agent persona and start chatting!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Agent selection
    agent_name = st.selectbox(
        "Select AI Agent",
        options=list(AGENTS.keys()),
        format_func=lambda x: f"{AGENTS[x]['icon']} {x}"
    )
    
    # Display agent description
    st.info(f"**{AGENTS[agent_name]['icon']} {agent_name}**\n\n{AGENTS[agent_name]['description']}")
    
    st.markdown("---")
    
    # AI Provider selection
    provider = st.selectbox(
        "Select AI Provider",
        ["Claude (Anthropic)", "ChatGPT (OpenAI)"]
    )
    
    # Get API keys from Streamlit secrets
    try:
        if provider == "Claude (Anthropic)":
            api_key = st.secrets["ANTHROPIC_API_KEY"]
            model = st.selectbox(
                "Select Model",
                ["claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
            )
        else:
            api_key = st.secrets["OPENAI_API_KEY"]
            model = st.selectbox(
                "Select Model",
                ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            )
        
        st.success("✅ API key loaded")
    
    except KeyError as e:
        st.error(f"⚠️ API key not found in secrets: {e}")
        st.info("Please check that your secret key name matches exactly in Streamlit settings")
        st.stop()
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                               help="Higher values make output more creative, lower values more focused")
        max_tokens = st.slider("Max Tokens", min_value=512, max_value=8192, value=4096, step=512,
                              help="Maximum length of the response")
    
    st.markdown("---")
    
    # Chat management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.messages:
                chat_export = "\n\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in st.session_state.messages
                    if isinstance(msg.get('content'), str)
                ])
                st.download_button(
                    label="Download",
                    data=chat_export,
                    file_name="chat_history.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # Display message count
    if st.session_state.get('messages'):
        msg_count = len([m for m in st.session_state.messages if m['role'] == 'user'])
        st.caption(f"💬 {msg_count} messages in conversation")
    
    st.markdown("---")
    st.markdown("### 📎 File Support")
    st.markdown("- **Images:** JPG, PNG, GIF, WebP\n- **Documents:** TXT, PDF, MD, Code files")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_agent" not in st.session_state:
    st.session_state.current_agent = agent_name

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Reset chat if agent changed
if st.session_state.current_agent != agent_name:
    if st.session_state.messages:
        st.warning("⚠️ Switching agents will clear your chat history")
        if st.button("Continue and clear chat"):
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.session_state.current_agent = agent_name
            st.rerun()
        if st.button("Cancel"):
            st.rerun()
        st.stop()
    else:
        st.session_state.current_agent = agent_name

# Display current agent at the top
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"### {AGENTS[agent_name]['icon']} Currently chatting with: **{agent_name}**")
with col2:
    if st.button("ℹ️ Agent Info"):
        st.info(AGENTS[agent_name]['system_prompt'])

st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle text content
        if isinstance(message.get("content"), str):
            st.markdown(message["content"])
        # Handle multimodal content (images + text)
        elif isinstance(message.get("content"), list):
            for content_block in message["content"]:
                if isinstance(content_block, dict):
                    if content_block.get("type") == "text":
                        st.markdown(content_block.get("text", ""))
                    elif content_block.get("type") == "image":
                        st.caption("🖼️ Image attached")
        
        # Display attached files
        if "files" in message:
            for file_info in message["files"]:
                with st.expander(f"📎 {file_info['name']}"):
                    if file_info['type'] == 'image':
                        st.image(file_info['data'])
                    else:
                        st.text(file_info['data'][:500] + "..." if len(file_info['data']) > 500 else file_info['data'])

# Show starter prompts only if chat is empty
if not st.session_state.messages:
    st.markdown("### 🚀 Get Started")
    st.markdown("Choose a starter prompt or type your own message below:")
    
    cols = st.columns(3)
    for idx, starter in enumerate(AGENTS[agent_name]["starters"]):
        with cols[idx]:
            if st.button(starter, key=f"starter_{idx}", use_container_width=True):
                # Set the starter as the prompt to be processed
                st.session_state.starter_prompt = starter
                st.rerun()

st.markdown("---")

# File upload integrated with message area
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown("### 💬 Your Message")
with col2:
    uploaded_files = st.file_uploader(
        "📎",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'pdf', 'md', 'py', 'js', 'html', 'css', 'json'],
        help="Attach files",
        label_visibility="visible"
    )

if uploaded_files:
    st.caption(f"📎 {len(uploaded_files)} file(s) attached - {', '.join([f.name for f in uploaded_files])}")

# Chat input
prompt = st.chat_input("Type your message here and press Enter...", key="chat_input")

# Check if we have a starter prompt to process
if "starter_prompt" in st.session_state:
    prompt = st.session_state.starter_prompt
    del st.session_state.starter_prompt

if prompt:
    # Prepare message content
    message_content = []
    attached_files = []
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = Path(uploaded_file.name).suffix.lower()
            mime_type = mimetypes.guess_type(uploaded_file.name)[0]
            
            # Handle images
            if mime_type and mime_type.startswith('image/'):
                image_data = encode_image(uploaded_file)
                
                if provider == "Claude (Anthropic)":
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_data
                        }
                    })
                else:  # OpenAI
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    })
                
                attached_files.append({
                    "name": uploaded_file.name,
                    "type": "image",
                    "data": uploaded_file.getvalue()
                })
            
            # Handle documents
            else:
                doc_content = read_document(uploaded_file)
                prompt = f"{prompt}\n\n[Attached document: {uploaded_file.name}]\n{doc_content}"
                
                attached_files.append({
                    "name": uploaded_file.name,
                    "type": "document",
                    "data": doc_content
                })
    
    # Add text to message content
    if message_content:
        message_content.append({"type": "text", "text": prompt})
    else:
        message_content = prompt
    
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": message_content
    }
    if attached_files:
        user_message["files"] = attached_files
    
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        if attached_files:
            for file_info in attached_files:
                if file_info['type'] == 'image':
                    st.image(file_info['data'], caption=file_info['name'], width=300)
                else:
                    st.caption(f"📄 {file_info['name']}")
    
    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if provider == "Claude (Anthropic)":
                # Claude API call
                client = Anthropic(api_key=api_key)
                
                # Convert messages to Claude format
                claude_messages = []
                for msg in st.session_state.messages:
                    claude_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=AGENTS[agent_name]["system_prompt"],
                    messages=claude_messages
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            else:
                # OpenAI API call
                client = OpenAI(api_key=api_key)
                
                # Convert messages to OpenAI format
                openai_messages = [
                    {"role": "system", "content": AGENTS[agent_name]["system_prompt"]}
                ]
                
                for msg in st.session_state.messages:
                    openai_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                stream = client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your API key configuration and try again.")

# Footer with helpful tips
if st.session_state.messages:
    with st.expander("💡 Tips for better conversations"):
        st.markdown("""
        - **Be specific:** The more details you provide, the better the response
        - **Upload files:** Attach images or documents for analysis
        - **Use agents:** Switch between specialized agents for different tasks
        - **Adjust settings:** Try different temperature values for varied responses
        - **Export chat:** Save your conversations for later reference
        """)
