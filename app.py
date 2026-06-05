import streamlit as st
from rag_pipeline import ask

st.title("College AI Assistant 🎓")
st.write("Ask anything about your college documents.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input at the bottom
query = st.chat_input("Ask a question...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.write(query)
    
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask(query)
        st.write(answer)
        
        # Show sources in expander
        with st.expander("View sources"):
            for i, chunk in enumerate(sources):
                st.markdown(f"**Source {i+1}:**")
                st.write(chunk)
                st.divider()
    
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
