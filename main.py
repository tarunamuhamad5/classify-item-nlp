from model import model_predict
import streamlit as st
import os

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    model = model_predict()
    st.title('Classification of Item Web App')
    
    heading = st.text_input('Text Heading of Item')
    section = st.selectbox('Section of Item',
                           ['for-sale','housing','services','community'])
    
    result =''
    
    if st.button('Classify Item'):
        result = model.predict(heading,section)
        
    st.success(result)


if __name__ == '__main__':
    main()