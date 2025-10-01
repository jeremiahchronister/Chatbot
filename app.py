import streamlit as st
from anthropic import Anthropic
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="Multi-AI Chatbot", page_icon="🤖", layout="wide")

# Define agent personas
AGENTS = {
    "General Assistant": {
        "icon": "🤖",
        "system_prompt": "You are a helpful, friendly, and knowledgeable AI assistant.",
        "description": "General purpose assistant for any task"
    },
    "Resume Helper": {
        "icon": "📝",
        "system_prompt": "You are an expert career coach and resume writer with 15+ years of experience. You help people create compelling resumes, optimize them for ATS systems, craft impactful bullet points using the STAR method, and provide tailored advice for different industries and career levels.",
        "description": "Expert help with resumes, cover letters, and career advice"
    },
    "Finance Advisor": {
        "icon": "💰",
        "system_prompt": "You are a knowledgeable financial advisor specializing in personal finance, investing, budgeting, and financial planning. You provide clear, actionable advice on saving, investing, retirement planning, tax strategies, and wealth management. You always remind users to consult with licensed professionals for personalized advice.",
        "description": "Personal finance, investing, and budgeting guidance"
    },
    "Healthcare Assistant": {
        "icon": "🏥",
        "system_prompt": "You are a healthcare information assistant with expertise in general health, wellness, nutrition, and medical information. You provide evidence-based health information, explain medical concepts clearly, and offer wellness tips. You always emphasize that your information is educational and users should consult healthcare professionals for medical advice, diagnosis, or treatment.",
        "description": "Health information, wellness tips, and medical education"
    },
    "Coding Mentor": {
        "icon": "💻",
        "system_prompt": "You are an experienced software engineer and coding mentor. You help with programming questions, debug code, explain concepts clearly, provide best practices, and guide learners through technical challenges. You write clean, well-documented code and explain your reasoning.",
        "description": "Programming help, code review, and technical guidance"
    },
    "Creative Writer": {
        "icon": "✍️",
        "system_prompt": "You are a creative writing coach and accomplished author. You help with storytelling, creative writing, content creation, editing, and developing engaging narratives. You provide constructive feedback, suggest improvements, and inspire creativity.",
        "description": "Creative writing, storytelling, and content creation"
    }
}

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
    
    # Clear chat button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot features specialized AI agents for different tasks. Select an agent above to get started!")

# Initialize chat history with agent context
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_agent" not in st.session_state:
    st.session_state.current_agent = agent_name

# Reset chat if agent changed
if st.session_state.current_agent != agent_name:
    st.session_state.messages = []
    st.session_state.current_agent = agent_name
    st.rerun()

# Display current agent at the top
st.markdown(f"### {AGENTS[agent_name]['icon']} Currently chatting with: **{agent_name}**")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if provider == "Claude (Anthropic)":
                # Claude API call
                client = Anthropic(api_key=api_key)
                
                # Convert messages to Claude format with system prompt
                claude_messages = [
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in st.session_state.messages
                ]
                
                with client.messages.stream(
                    model=model,
                    max_tokens=4096,
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
                
                # Convert messages to OpenAI format with system message
                openai_messages = [
                    {"role": "system", "content": AGENTS[agent_name]["system_prompt"]}
                ] + [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]
                
                stream = client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
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
