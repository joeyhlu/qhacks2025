import streamlit as st

def main():
    # Page set up
    home_page = st.Page(
        page="pages/home.py", 
        title="Home", 
        icon="🏠",
        default=True,
    )

    video_page = st.Page(
        page="pages/video.py", 
        title="Application", 
        icon="📷"
    )


    page = st.navigation(pages=[home_page, video_page])

    st.logo("tldr.png")
    st.sidebar.text("")

    page.run()



main()


 
