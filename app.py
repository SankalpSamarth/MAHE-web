import streamlit as st
from test import ask

st.title("College AI Assistant 🎓")
st.write("Ask anything about your college documents.")

query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking..."):
        answer = ask(query)
    st.subheader("Answer:")
    st.write(answer)
