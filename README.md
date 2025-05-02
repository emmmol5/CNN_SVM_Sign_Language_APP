A machine learning pipeline that classifies Norwegian Sign Language (NSL) videos into 720 healthcare-related terms using a pretrained MobileNetV2 feature extractor and an SVM classifier. 

Features:
Frame sampling + preprocessing for short sign videos, 
Feature extraction using MobileNetV2 (frozen),
Classification using Support Vector Machine (SVM),
Label encoding and decoding,
Web application interface built with Streamlit,
Dataset includes original + 10 augmentations per class
