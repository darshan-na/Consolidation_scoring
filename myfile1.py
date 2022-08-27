import requests
import streamlit as st
from streamlit_lottie import st_lottie
import base64
from PIL import Image


#st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Data Professor</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">YouTube</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Twitter</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)





def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#--load assets--
lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_izvmskti.json")

lottie_new = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_pzprncar.json")

#---Header Section---
with st.container():
    def header(url):
        st.markdown(f'<p style="background-color:#00FF00;color:#000000;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

    def header2(url):
        st.markdown(f'<p style="background-color:#FF1493;color:#000000;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

    header2("Consolidation Scoring of Lung")
    
    header("Welcome to CAD")
    
    st.header("Lung Segmentation and Ribs supression")
    #st.write("welcome")
    
    st.write("[Explore](https://pubmed.ncbi.nlm.nih.gov/22606675/)")

#--work---
with st.container():
    st.write("---")
    def header1(url):
        st.markdown(f'<p style="background-color:#00FF00;color:#000000;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
    header1("How can we help you?")
    st.write(" A number of computer-aided diagnosis (CAD) systems for consolidation scoring on chest radiographs have been proposed to improve detection accuracy . For current CAD schemes nodule detection using chest radiographs presents a major challenge because ribs often obscure lessions. Because these bony structures tend to cause CAD systems to produce false positives, for clinical applications, CAD systems have the disadvantage of lower sensitivity and specificity. A recent study showed that most lung cancer lesions missed on frontal chest radiographs are located behind the ribs, and that inspection of a soft-tissue image can improve human detection ")
    left_column, right_column = st.columns(2)
    with left_column:
        #st.header("What we do")
        st_lottie(lottie_new, height=200, key="new")
        #st.write("##")
    with right_column:
        st_lottie(lottie_coding, height=200, key="coding")

with st.expander("-> RIBS SUPPRESION ", expanded=False):
      st.write(
          """    
  - RIBS SUPRESSION:
      - Clahe method for enhanchment of image:
      - used to improve contrast of the image enhanched so that histogram output matches the histogram of the entire image .
      - to maintain good amount of accuracy ,the lung feild carefully extracted from the image manually by earsing the other parts such as spinal caord.
      - Since , instesity of pixels is discretely distributed on the image called the entire image.
      - ribcage shadow suppresion ,we have to first obtain the ribcage for the which is the gabour filtreis introduced.
      - Gabor filtre:
      - A linaer filtre used to texture anaylsis ,they play important rule for extracting ribs.
      - They highlight the features or texture of images along with which it is oriented and suppressed rest all the features of image.
          """
      )
      st.write("Original Image")
      img = Image.open("pic2.png")
      st.image(img)
      st.write("---")
      st.write("Rib Suppressed Image")
      img = Image.open("pic1.png")
      st.image(img)

with st.expander("-> LUNG Segmentation ", expanded=False):
      st.write(
          """   
      - LUNG SEGMENTATION:
      - Data set consisted of 800 images,out of which 704 had a ground truth marked by the doctor.
      - Entire data was divided into three division  which include :  train,test and validation .
      - Binary semantic segmentation was employed with the help of universal architecture (U-Net Model).
      - Model was intsialised with zero centric random weights.
      - Model was trained over total of 25 epoches and we achived a dice score  of 80% on the training data and 76% on the test data.
      - In the future we aim to increase the dice score of the model through hyper parameter tunning.""")
      st.write("Original Image")
      img = Image.open("CHNCXR_0003_0.png")
      st.image(img)
      st.write("Original Image")
      img = Image.open("CHNCXR_0468_1.png")
      st.image(img)
      st.write("Original Image")
      img = Image.open("CHNCXR_0655_1.png")
      st.image(img)
with st.container():
    with st.expander("-> Unet Model ", expanded=False):
      img = Image.open("unet.png")
      st.image(img)
      
with st.container():
    with st.expander("-> Consolidation Scoring ", expanded=False):
      st.write('''
          Lung consolidation occurs when the air that usually fills the small airways in your lungs is replaced with something else. Depending on the cause, the air may be replaced with: a fluid, such as pus, blood, or water
a solid, such as stomach contents or cells.
      
      -Consolidation can be used to prevent early stages of :
      - Pneumonia
      - Pulmonary edema
      - Pulmonary hemorrhage
      - Lung cancer
      - Aspiration
            ''')
      st.write("original image")
      img = Image.open("cs.jpg")
      st.image(img)
      st.write("actual consolidation seen on the lower left lobe")
      img = Image.open("cs2.png")
      st.image(img)
      

    
st.write("Contributers-")
st.write('''Darshan NA
            ''')

        #st.write("hello")


       

           

        


