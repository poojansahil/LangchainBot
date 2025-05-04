import streamlit as st
import os
from datetime import datetime
from PIL import Image
from io import BytesIO

# Imports
try:
    import google.generativeai as genai
    from openai import OpenAI
    from huggingface_hub import InferenceClient

    from langchain.memory import ConversationBufferMemory
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.info("Make sure to install dependencies: pip install streamlit google-generativeai openai huggingface_hub langchain langchain_google_genai langchain_openai")
    st.stop()

# Page config
st.set_page_config(
    page_title="Gemini + OpenAI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Session state init
for key in ["messages", "conversation_history", "gemini_api_key", "openai_api_key", "huggingface_api_key", "chat_memory"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else "" if "api_key" in key or key == "conversation_history" else ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# App header
st.title("🤖 Gemini Chatbot + OpenAI Summary + Hugging Face Image")
st.subheader("Chat using Gemini, summarize with OpenAI, generate images via Hugging Face")

# Sidebar
with st.sidebar:
    st.header("🔑 API Keys")

    # Gemini Key
    gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key_input")
    if gemini_key:
        st.session_state.gemini_api_key = gemini_key

    # OpenAI Key
    openai_key = st.text_input("OpenAI API Key (for summary)", type="password", key="openai_key_input")
    if openai_key:
        st.session_state.openai_api_key = openai_key

    # Hugging Face Key
    hf_key = st.text_input("Hugging Face API Key (for image generation)", type="password", key="hf_key_input")
    if hf_key:
        st.session_state.huggingface_api_key = hf_key

    # Prompt for image
    st.markdown("---")
    st.header("🖼️ Generate Image")
    image_prompt = st.text_input("Enter image prompt")

    if st.button("Generate Image"):
        if not st.session_state.huggingface_api_key:
            st.warning("Please enter a Hugging Face API key.")
        elif not image_prompt.strip():
            st.warning("Please enter a valid prompt.")
        else:
            try:
                with st.spinner("Generating image..."):
                    client = InferenceClient(
                        provider="nebius",
                        api_key=st.session_state.huggingface_api_key
                    )
                    image = client.text_to_image(
                        prompt=image_prompt,
                        model="black-forest-labs/FLUX.1-dev"
                    )
                    st.image(image, caption=image_prompt)
            except Exception as e:
                st.error(f"Image generation error: {e}")

    # Summarize
    st.markdown("---")
    st.header("📝 Summary")
    end_button_disabled = not st.session_state.openai_api_key or len(st.session_state.messages) <= 1
    if st.button("End Chat and Summarize", disabled=end_button_disabled):
        with st.spinner("Summarizing..."):
            formatted_convo = ""
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_convo += f"{role}: {msg['content']}\n\n"

            summary_prompt = PromptTemplate(
                input_variables=["conversation"],
                template="""
                You are a highly skilled AI assistant tasked with summarizing conversations.
                Below is a conversation between a user and an AI assistant.
                Please provide:
                1. A concise summary in under 150 words.
                2. Sentiment analysis (positive, negative, neutral, or mixed).

                Conversation:
                {conversation}

                Summary:
                """
            )
            try:
                openai_llm = ChatOpenAI(
                    temperature=0.3,
                    api_key=st.session_state.openai_api_key
                )
                summary_chain = LLMChain(llm=openai_llm, prompt=summary_prompt)
                summary = summary_chain.run(conversation=formatted_convo)

                # Display summary
                st.session_state.messages.append({"role": "assistant", "content": f"**Conversation Summary:**\n\n{summary}"})
                st.success("✅ Summary generated")

                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"conversation_{timestamp}.txt"
                with open(file_name, "w") as f:
                    f.write(formatted_convo)
                    f.write("\n\n--- SUMMARY ---\n\n")
                    f.write(summary)
                with open(file_name, "rb") as f:
                    st.download_button("Download Summary", f, file_name, mime="text/plain")

            except Exception as e:
                st.error(f"OpenAI error: {e}")

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    if not st.session_state.gemini_api_key:
        st.error("Please enter your Gemini API key.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history += f"User: {prompt}\n"

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gemini is thinking..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=st.session_state.gemini_api_key,
                    temperature=0.7,
                    convert_system_message_to_human=True
                )

                chat_prompt = PromptTemplate(
                    input_variables=["chat_history", "user_input"],
                    template="""
                    You are a helpful and friendly AI assistant.
                    Conversation history:
                    {chat_history}

                    User: {user_input}
                    Assistant:
                    """
                )

                conversation_chain = LLMChain(
                    llm=llm,
                    prompt=chat_prompt,
                    memory=st.session_state.chat_memory,
                    verbose=False
                )

                response = conversation_chain.predict(user_input=prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.conversation_history += f"Assistant: {response}\n"
            except Exception as e:
                st.error(f"Gemini error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
