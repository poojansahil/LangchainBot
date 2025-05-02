import streamlit as st
import os
from datetime import datetime

# Try import statements with specific error handling
try:
    # First try importing the core packages
    import google.generativeai as genai
    from openai import OpenAI
    
    # Now attempt to import the full set of required packages
    try:
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_openai import ChatOpenAI
    except ImportError:
        st.error("Unable to import LangChain modules. Please check your installation.")
        st.stop()
        
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.info("Make sure all dependencies are installed: `pip install langchain langchain_google_genai langchain_openai google-generativeai openai`")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Chatbot with Gemini & OpenAI", 
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# App title and description
st.title("🤖 Chatbot with Gemini & OpenAI")
st.subheader("Chat with Gemini, Summarize with OpenAI")

# Display instructions for first-time users
if not st.session_state.messages:
    st.info("""
    ### How to use this chatbot:
    1. Enter your **Gemini API key** in the sidebar to start chatting
    2. When you're ready to end the conversation, enter your **OpenAI API key** in the sidebar
    3. Click **End Chat and Summarize** to get a summary and sentiment analysis
    4. Download your conversation if desired
    """)

# API key input area in sidebar
with st.sidebar:
    st.header("API Keys")
    st.markdown("Enter your API keys below. These are stored only for the current session and are not saved permanently.")
    
    # Gemini API key input
    gemini_key = st.text_input("Enter Gemini API Key", type="password", key="gemini_key_input")
    if gemini_key:
        st.session_state.gemini_api_key = gemini_key
    
    # Display API key instructions
    if not st.session_state.gemini_api_key:
        st.info("Please enter your Gemini API key above to start chatting.")
    
    # OpenAI API key input
    openai_key = st.text_input("Enter OpenAI API Key (for summary)", type="password", key="openai_key_input")
    if openai_key:
        st.session_state.openai_api_key = openai_key
    
    # Instructions for OpenAI API key
    if not st.session_state.openai_api_key:
        st.info("You'll need to enter an OpenAI API key to generate a summary when ending the chat.")

    # Add End Chat button in sidebar
    end_button_disabled = not st.session_state.openai_api_key or len(st.session_state.messages) <= 1
    
    if st.button("End Chat and Summarize", type="primary", disabled=end_button_disabled):
        if len(st.session_state.messages) > 1:  # Check if there's a conversation to summarize
            with st.spinner("Generating summary..."):
                # Prepare conversation for summary
                formatted_conversation = ""
                for msg in st.session_state.messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    formatted_conversation += f"{role}: {msg['content']}\n\n"
                
                # Create summary template
                summary_template = """
                You are a highly skilled AI assistant tasked with summarizing conversations.
                Below is a conversation between a user and an AI assistant.
                Please provide:
                1. A concise summary of the conversation in under 150 words.
                2. A brief sentiment analysis of the conversation (positive, negative, neutral, or mixed).

                Conversation:
                {conversation}

                Summary:
                """
                
                summary_prompt = PromptTemplate(
                    input_variables=["conversation"],
                    template=summary_template
                )
                
                # Initialize OpenAI LLM
                try:
                    openai_llm = ChatOpenAI(
                        temperature=0.3,
                        api_key=st.session_state.openai_api_key
                    )
                    
                    # Create summarization chain
                    summary_chain = LLMChain(
                        llm=openai_llm,
                        prompt=summary_prompt
                    )
                    
                    # Generate summary
                    summary_result = summary_chain.run(conversation=formatted_conversation)
                    
                    # Display summary in main area
                    st.session_state.messages.append({"role": "assistant", "content": f"**Conversation Summary:**\n\n{summary_result}"})
                    
                    # Option to download conversation
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    conversation_file = f"conversation_{timestamp}.txt"
                    with open(conversation_file, "w") as f:
                        f.write(formatted_conversation)
                        f.write("\n\n--- SUMMARY ---\n\n")
                        f.write(summary_result)
                    
                    with open(conversation_file, "r") as f:
                        st.sidebar.download_button(
                            label="Download Conversation",
                            data=f,
                            file_name=conversation_file,
                            mime="text/plain"
                        )
                        
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
        else:
            st.warning("Not enough conversation to summarize.")
    
    st.divider()
    st.caption("This chatbot uses Gemini 1.5 Flash for chat and OpenAI for summarization.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Check if Gemini API key is provided
    if not st.session_state.gemini_api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar to start chatting.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history += f"User: {prompt}\n"
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response using Gemini
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Initialize Gemini LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=st.session_state.gemini_api_key,
                    temperature=0.7,
                    convert_system_message_to_human=True
                )
                
                # Create chat prompt template
                chat_template = """
                You are a helpful and friendly AI assistant. 
                The conversation history is provided below for context.
                
                Conversation history:
                {chat_history}
                
                User: {user_input}
                Assistant:
                """
                
                chat_prompt = PromptTemplate(
                    input_variables=["chat_history", "user_input"],
                    template=chat_template
                )
                
                # Create conversation chain
                conversation_chain = LLMChain(
                    llm=llm,
                    prompt=chat_prompt,
                    memory=st.session_state.chat_memory,
                    verbose=True
                )
                
                # Get response
                response = conversation_chain.predict(user_input=prompt)
                
                # Display and save response
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.conversation_history += f"Assistant: {response}\n"
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})