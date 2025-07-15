import gradio as gr
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from detection_service import DetectionService
from analysis_service import AnalysisService

# Initialize services - will be done dynamically
detector = None
analyzer = None

# Global settings storage
settings = {
    "yolo_endpoint": os.getenv("YOLO_ENDPOINT", "https://your-yolo-endpoint.com/predict"),
    "yolo_api_key": os.getenv("YOLO_API_KEY", "your-yolo-api-key-here"),
    "qwen_endpoint": os.getenv("QWEN_ENDPOINT", "https://your-qwen-endpoint.com/v1/chat/completions"),
    "qwen_api_key": os.getenv("QWEN_API_KEY", "your-qwen-api-key-here")
}

def update_settings(yolo_endpoint, yolo_api_key, qwen_endpoint, qwen_api_key):
    """Update API settings and reinitialize services"""
    global detector, analyzer, settings
    
    # Update settings
    settings["yolo_endpoint"] = yolo_endpoint
    settings["yolo_api_key"] = yolo_api_key
    settings["qwen_endpoint"] = qwen_endpoint
    settings["qwen_api_key"] = qwen_api_key
    
    # Update environment variables
    os.environ["YOLO_ENDPOINT"] = yolo_endpoint.replace("/predict", "") if yolo_endpoint else ""
    os.environ["YOLO_API_KEY"] = yolo_api_key if yolo_api_key else ""
    os.environ["QWEN_ENDPOINT"] = qwen_endpoint if qwen_endpoint else ""
    os.environ["QWEN_API_KEY"] = qwen_api_key if qwen_api_key else ""
    
    # Force reinitialize services
    detector = None
    analyzer = None
    
    return "Settings updated successfully! You can now use the analysis features."

def get_detector():
    """Get detector instance with current environment variables"""
    global detector
    if detector is None:
        detector = DetectionService()
    return detector

def get_analyzer():
    """Get analyzer instance with current environment variables"""
    global analyzer
    if analyzer is None:
        analyzer = AnalysisService()
    return analyzer

def process_image(image):
    """Process single image for traffic analysis"""
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Get detections
        detections = get_detector().detect_objects(pil_image)
        
        # Draw detections on image
        annotated_image = get_detector().draw_detections(pil_image, detections)
        
        # Get analysis only if we have detections or if API keys are configured
        if detections or (os.getenv("QWEN_ENDPOINT") and os.getenv("QWEN_API_KEY")):
            analysis = get_analyzer().analyze_traffic_scene(pil_image, detections)
        else:
            analysis = {"content": "No analysis available - configure API keys to enable AI analysis"}
        
        # Format results
        results_text = format_analysis_results(detections, analysis)
        
        return annotated_image, results_text
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def process_video(video_path):
    """Process video for traffic analysis (sample frames)"""
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames (every 30 frames or 1 second)
        sample_interval = max(1, int(fps))
        results = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                
                # Process frame
                detections = get_detector().detect_objects(pil_frame)
                analysis = get_analyzer().analyze_traffic_scene(pil_frame, detections)
                
                # Store result
                results.append({
                    "frame": frame_count,
                    "timestamp": frame_count / fps,
                    "detections": detections,
                    "analysis": analysis
                })
                
                # Limit to 5 samples for demo
                if len(results) >= 5:
                    break
                    
            frame_count += 1
        
        cap.release()
        
        # Get a representative frame for display
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)  # Middle frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            detections = get_detector().detect_objects(pil_frame)
            annotated_frame = get_detector().draw_detections(pil_frame, detections)
        else:
            annotated_frame = None
        
        # Format video results
        video_results = format_video_results(results)
        
        return annotated_frame, video_results
        
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

def format_analysis_results(detections, analysis):
    """Format analysis results for display"""
    result_text = "## Traffic Analysis Results\n\n"
    
    # Detection summary - count objects by type
    result_text += "### Number of detected vehicles:\n"
    if detections:
        object_counts = {}
        for detection in detections:
            obj_type = detection['class'].title()
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        for obj_type, count in object_counts.items():
            result_text += f"- {obj_type}: {count}\n"
    else:
        result_text += "- No traffic objects detected\n"
        result_text += "- *Note: Configure YOLO API keys in Settings to enable real-time detection*\n"
    
    # Analysis results
    result_text += "\n### AI-powered traffic report:\n"
    analysis_content = analysis.get('content', 'Analysis not available')
    result_text += analysis_content
    
    return result_text

def format_video_results(results):
    """Format video analysis results"""
    if not results:
        return "No analysis results available"
    
    result_text = "## Video Traffic Analysis Results\n\n"
    
    for i, result in enumerate(results, 1):
        result_text += f"### Frame {i} (t={result['timestamp']:.1f}s)\n"
        
        # Detection count
        detections = result['detections']
        if detections:
            detection_counts = {}
            for detection in detections:
                cls = detection['class']
                detection_counts[cls] = detection_counts.get(cls, 0) + 1
            
            for cls, count in detection_counts.items():
                result_text += f"- {cls.title()}: {count}\n"
        else:
            result_text += "- No objects detected\n"
        
        # Analysis
        analysis = result['analysis']
        result_text += f"- **Flow:** {analysis.get('traffic_flow', 'Unknown')}\n"
        result_text += f"- **Safety:** {analysis.get('safety_concerns', 'OK')}\n\n"
    
    return result_text

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Smart Traffic Report") as demo:
        gr.Markdown("# Smart Traffic Report")
        gr.Markdown("Upload an image or video to analyze traffic conditions using YOLO detection and Qwen2.5-VL analysis.")
        
        with gr.Tabs():
            with gr.TabItem("Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload Traffic Image")
                        image_button = gr.Button("Analyze Image", variant="primary")
                    
                    with gr.Column():
                        image_output = gr.Image(label="Detection Results")
                        image_analysis = gr.Markdown(label="Analysis Results")
                
                image_button.click(
                    process_image,
                    inputs=[image_input],
                    outputs=[image_output, image_analysis]
                )
            
            with gr.TabItem("Video Analysis"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Traffic Video")
                        video_button = gr.Button("Analyze Video", variant="primary")
                    
                    with gr.Column():
                        video_frame_output = gr.Image(label="Sample Frame with Detections")
                        video_analysis = gr.Markdown(label="Video Analysis Results")
                
                video_button.click(
                    process_video,
                    inputs=[video_input],
                    outputs=[video_frame_output, video_analysis]
                )
            
            with gr.TabItem("Endpoint Configuration"):
                gr.Markdown("## API Configuration")
                gr.Markdown("Configure your YOLO and Qwen2.5-VL model endpoints and API keys.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### YOLO Detection Settings")
                        yolo_endpoint_input = gr.Textbox(
                            label="YOLO Endpoint",
                            placeholder="https://your-yolo-endpoint.com/predict",
                            value=settings["yolo_endpoint"],
                            info="Full endpoint URL including /predict"
                        )
                        yolo_key_input = gr.Textbox(
                            label="YOLO API Key",
                            placeholder="your-yolo-api-key-here",
                            value=settings["yolo_api_key"],
                            type="password",
                            info="Your YOLO API authentication key"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Qwen2.5-VL Analysis Settings")
                        qwen_endpoint_input = gr.Textbox(
                            label="Qwen Endpoint",
                            placeholder="https://your-qwen-endpoint.com/v1/chat/completions",
                            value=settings["qwen_endpoint"],
                            info="Full endpoint URL including /v1/chat/completions"
                        )
                        qwen_key_input = gr.Textbox(
                            label="Qwen API Key",
                            placeholder="your-qwen-api-key-here",
                            value=settings["qwen_api_key"],
                            type="password",
                            info="Your Qwen API authentication key"
                        )
                
                with gr.Row():
                    settings_button = gr.Button("Save Settings", variant="primary", size="lg")
                    settings_status = gr.Markdown("")
                
                settings_button.click(
                    update_settings,
                    inputs=[yolo_endpoint_input, yolo_key_input, qwen_endpoint_input, qwen_key_input],
                    outputs=[settings_status]
                )
                
                gr.Markdown("### Instructions:")
                gr.Markdown("1. **YOLO Endpoint**: Enter your YOLOv8 detection service URL ending with `/predict`")
                gr.Markdown("2. **YOLO API Key**: Enter your authentication key for the YOLO service")
                gr.Markdown("3. **Qwen Endpoint**: Enter your Qwen2.5-VL service URL ending with `/v1/chat/completions`")
                gr.Markdown("4. **Qwen API Key**: Enter your authentication key for the Qwen service")
                gr.Markdown("5. **Save Settings**: Click to apply the new configuration")
                
                gr.Markdown("### Note:")
                gr.Markdown("Settings are applied immediately and will be used for all subsequent analyses. The demo includes fallback detection data when APIs are unavailable.")
        
        gr.Markdown("### Instructions:")
        gr.Markdown("1. **Configure APIs**: Go to Settings tab to set up your model endpoints")
        gr.Markdown("2. **For Images**: Upload a traffic scene image to get real-time analysis")
        gr.Markdown("3. **For Videos**: Upload a traffic video to analyze multiple frames")
        gr.Markdown("4. **Results**: View detected objects and AI-generated traffic insights")
        
        gr.Markdown("---")
        gr.Markdown("*Powered by YOLOv8 for object detection and Qwen2.5-VL for scene analysis*")
    
    return demo

if __name__ == "__main__":
    # Check if API keys are set
    det = get_detector()
    ana = get_analyzer()
    if not det.api_key or not ana.api_key:
        print("Warning: API keys not found. Please set environment variables:")
        print("- YOLO_ENDPOINT and YOLO_API_KEY")
        print("- QWEN_ENDPOINT and QWEN_API_KEY")
    
    # Launch the app
    demo = create_interface()
    try:
        demo.launch(
            server_name='127.0.0.1',
            server_port=8080,
            share=True,
            show_error=True,
            prevent_thread_lock=False,
            allowed_paths=["/tmp"]
        )
    except Exception as e:
        print(f"Error launching Gradio app: {e}")
        # Fallback launch with minimal configuration
        demo.launch(
            server_name='127.0.0.1',
            server_port=8080,
            share=True
        )