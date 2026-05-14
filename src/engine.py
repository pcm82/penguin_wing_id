"""
AI Engine for Penguin Wing Identification.
Handles model loading and feature extraction using MegaDescriptor.
"""

import timm
import streamlit as st
from wildlife_tools.features import DeepFeatures

@st.cache_resource
def load_biometric_engine():
    """
    Loads the MegaDescriptor model once and caches it in memory.
    """
    model_name = 'hf-hub:BVRA/MegaDescriptor-T-224'
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    # num_workers=0 is required for Windows stability in Streamlit
    extractor = DeepFeatures(model, num_workers=0)
    return model, extractor

def get_database_fingerprints(dataset, extractor):
    """
    Generates biometric fingerprints for the entire library of wing photos.
    """
    return extractor(dataset)
