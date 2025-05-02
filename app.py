import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Langchain Chatbot", page_icon="💬")

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
st.title("🤖 Langchain Chatbot")
st.subheader("Chat with Gemini, Summarize with OpenAI")

# API key input area in sidebar
with st.sidebar:
    st.header("API Keys")
    
    # Gemini API key input
    gemini_key = st.text_input("Enter Gemini API Key", type="password", key="gemini_key_input")
    if gemini_key:
        st.session_state.gemini_api_key = gemini_key
    
    # OpenAI API key input
    openai_key = st.text_input("Enter OpenAI API Key (for summary)", type="password", key="openai_key_input")
    if openai_key:
        st.session_state.openai_api_key = openai_key
    
    # Add End Chat button in sidebar
    if st.button("End Chat and Summarize", type="primary", disabled=not st.session_state.openai_api_key):
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
                    
                    # Create summarization chain using newer langchain interface
                    summary_chain = (
                        {"conversation": lambda x: x}
                        | summary_prompt
                        | openai_llm
                        | StrOutputParser()
                    )
                    
                    # Generate summary
                    summary_result = summary_chain.invoke(formatted_conversation)
                    
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
        st.warning("Please enter your Gemini API key in the sidebar.")
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
                
                # Create conversation chain using the newer langchain interface
                conversation_chain = (
                    {"chat_history": lambda x: st.session_state.chat_memory.load_memory_variables({})["chat_history"],
                     "user_input": lambda x: x}
                    | chat_prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Get response
                response = conversation_chain.invoke(prompt)
                
                # Update memory
                st.session_state.chat_memory.save_context(
                    {"input": prompt},
                    {"output": response}
                )
                
                # Display and save response
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.conversation_history += f"Assistant: {response}\n"
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})