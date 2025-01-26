import streamlit as st

def main():
    # Page set up
    home_page = st.Page(
        page="camera/pages/home.py", 
        title="Home", 
        icon="🏠",
        default=True,
    )

    video_page = st.Page(
        page="camera/pages/video.py", 
        title="Application", 
        icon="📷"
    )


    page = st.navigation(pages=[home_page, video_page])
    st.sidebar.text("")

    page.run()



main()


 
