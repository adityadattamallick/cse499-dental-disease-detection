from ultralytics import YOLO
import streamlit as st
import settings
import cv2

def load_model(model_path):
    """
    Load a YOLO model from the specified path.
    
    Args:
        model_path (str): Path to the YOLO model file (e.g., 'yolov8n.pt')
    
    Returns:
        YOLO: Loaded YOLO model object ready for inference
    """
    # Initialize and load the YOLO model from the given path
    model = YOLO(model_path)
    return model