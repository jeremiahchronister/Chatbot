import streamlit as st
from anthropic import Anthropic
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="Multi-AI Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 Multi-AI Chatbot")
st.markdown("Chat with Claude or ChatGPT")

# Sidebar for API configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AI Provider selection
    provider = st.selectbox(
        "Select AI Provider",
        ["Claude (Anthropic)", "ChatGPT (OpenAI)"]
    )
    
    # API Key input
    if provider == "Claude (Anthropic)":
        api_key = st.text_input("Enter Claude API Key", type="password", key="claude_key")
        model = st.selectbox(
            "Select Model",
            ["claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
        )
    else:
        api_key = st.text_input("Enter OpenAI API Key", type="password", key="openai_key")
        model = st.selectbox(
            "Select Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        )
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot supports both Claude and ChatGPT APIs. Enter your API key above to get started.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Check if API key is provided
    if not api_key:
        st.error("Please enter an API key in the sidebar.")
        st.stop()
    
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
                
                # Convert messages to Claude format
                claude_messages = [
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in st.session_state.messages
                ]
                
                with client.messages.stream(
                    model=model,
                    max_tokens=4096,
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
            st.info("Please check your API key and try again.")
