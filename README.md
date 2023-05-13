[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

<div id="top"></div>

<div align="center">
  <a href="https://github.com/vsuraj25">
    <img src="https://img.icons8.com/dusk/64/pets.png" alt="Logo" width="80" height="80"/> 
  </a>

    
<h3 align="center">Pets Image Classification(Dog and Cats)</h3>

 <p align="center">
    Deep Learning Project 
    <br />
    <a href="https://github.com/vsuraj25"><strong>Explore my Repositories. »</strong></a>
    <br />
    <br />
    <a href="#intro">Introduction</a>
    ·
    <a href="#data"> Data Information</a>
    ·
    <a href="#contact">Contact</a>
  </p>
</div>

<div id="intro"></div>
<!-- ABOUT THE PROJECT -->

## **Introduction**
* This is an end to end implementation for classification of images of dogs and cats. Deep learning technique with transfer learning is implied for buliding the classification model. The Imagenet dataset is used for transfer learing. Applications such as DVC, MLFlow and Dagshub are used for tracking and monitoring the data and models. The prediction interface is achieved using Steamlit.

## **Deployed app**
[![App Screenshot](https://user-images.githubusercontent.com/55409076/238143936-811f798f-62d0-4905-b18f-d491071010fc.png)](https://pet-image-classification.onrender.com/)

[Deployed app link](https://pet-image-classification.onrender.com/)

 
<div id="data"></div>
<!-- USAGE EXAMPLES -->

## **Dataset Information**

* Download the original dataset here : 
  [Cats and dogs image dataset](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip)

 
* The dataset is available in compressed format. 
* It contains 12500 images of cats and 12500 images of cats. 
* Each image is of dimension 224 x 224 x 3, satisfying the VGG16 model image input size requirements.
* These images are 3 channeled images i.e. colour images. 
* Original number images 25000

## **Steps**

1. Ingest the data and extract the zip file to save the cats and dogs images.
2. Initialize VGG16 as base model using the imagenet dataset and deactivate the desired layers for transfer learning.
3. Train,test and validate the model using the downloaded image dataset.
4. Keep track of callbacks and checkpoints usinf tensorboard and ModelCheckpoints.
5. Evaluate the model.
6. Log the model parameters and metrics in MLFlow for model monitiring.
7. Create a prediction service using the Streamlit on the best model. 
8. Deploy the application on cloud.

<p align="right">(<a href="#top">back to top</a>)</p> 

<!-- USAGE EXAMPLES -->
## **Project Architecture**

[![Project Architecture](https://user-images.githubusercontent.com/55409076/238143866-16ba7f37-17f0-4e1f-9587-c659ceff06f7.png)


## **Requirements**
* Python 3.7
* Numpy
* Tensorflow
* Streamlit
* Pytest
* Tox
* DVC
* MLFlow
* Checkout requirements.txt for more information.

## **Technologies used**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dataversioncontrol&logoColor=white)
![MLFlow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Keras](https://img.shields.io/badge/Keras-D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Yaml](https://img.shields.io/badge/YAML-CB171E.svg?style=for-the-badge&logo=YAML&logoColor=white)
![Json](https://img.shields.io/badge/JSON-000000.svg?style=for-the-badge&logo=JSON&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC.svg?style=for-the-badge&logo=Pytest&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=for-the-badge&logo=Docker&logoColor=white)


## **Tools used**
![Visual Studio Code](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)
![MLFlow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=for-the-badge&logo=Docker&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![Conda](https://img.shields.io/badge/Anaconda-44A833.svg?style=for-the-badge&logo=Anaconda&logoColor=white)

<!-- CONTACT -->
<div id="contact"></div>

## **Contact**
[![Suraj Verma | LinkedIn](https://img.shields.io/badge/Suraj_Verma-eeeeee?style=for-the-badge&logo=linkedin&logoColor=ffffff&labelColor=0A66C2)][reach_linkedin]
[![Suraj Verma | G Mail](https://img.shields.io/badge/sv255255-eeeeee?style=for-the-badge&logo=gmail&logoColor=ffffff&labelColor=EA4335)][reach_gmail]
[![Suraj Verma | G Mail](https://img.shields.io/badge/My_Portfolio-eeeeee?style=for-the-badge)][reach_gmail]

[reach_linkedin]: https://www.linkedin.com/in/suraj-verma-982b31157/
[reach_gmail]: mailto:sv255255@gmail.com?subject=Github


<p align="right">(<a href="#top">back to top</a>)</p>



