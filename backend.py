import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    pipeline,
    DistilBertTokenizer, 
    DistilBertForSequenceClassification
)
from ultralytics import YOLO
import timm
from torchvision import transforms
import logging
import os
import re
import gc
from ctransformers import AutoModelForCausalLM

class Config:
    # Model Paths
    YOLO_PATH = "electronics_type_classifier/runs/detect/train4/weights/best.pt"
    CLASSIFIER_PATH = "condition_classifier/defect_classifier_v1.pth"
    NLP_PATH = "thalleous/EcoScan-NLP"
    LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Token (Only needed for Gemma)
    HF_TOKEN = "" 
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    
    # VISUAL CLASSIFIER CLASSES (What your ResNet knows)
    IMG_CLASSES = [
        'laptop_normal', 'laptop_crack', 'laptop_fades', 'laptop_lines', 
        'laptop_spot', 'phone_dead_pixel', 'phone_scratch', 'phone_crack', 'phone_normal'
    ]
    
    # NLP LABELS
    NLP_LABELS = [
        'Power_Failure', 'Battery_Charging', 'Display_Visual', 'Audio_Sound',
        'Overheating_Thermal', 'Connectivity_Signal', 'Water_Liquid_Damage',
        'Mechanical_Motor', 'Input_Controls', 'Software_Error', 'Data_Storage',
        'Sensor_Accuracy'
    ]

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- COMPONENT 1: LLM ADVISOR ---
class DeviceAdvisorLLM:
    def __init__(self, model_id, hf_token):
        logger.info(f"Loading LLM via ctransformers...")
        
        # ctransformers handles the download and loading in one step!
        self.llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            model_type="llama",
            context_length=2048,
            gpu_layers=0  # Force CPU usage
        )
        logger.info("LLM Loaded successfully.")

    def generate_recommendation(self, device_type, visual_condition, nlp_issues):
        # 1. Clean Inputs
        visual_clean = visual_condition.replace("_", " ").title()
        issues_clean = ", ".join([x.replace("_", " ") for x in nlp_issues]) if nlp_issues else "None"

        # 2. Prompt (Same as before)
        prompt = f"""<|system|>
You are a technical diagnostic tool. Output the status in the exact format shown. 
Do not provide repair instructions or steps. Keep descriptions brief.
</s>
<|user|>
Device: Smartphone
Visual: Screen Crack
Internal: Touch Issue

Response:
Diagnosis: Cracked Screen Digitizer
Severity: Medium
Action: Replace Screen
Reasoning: Physical damage is interfering with touch sensors.
</s>
<|user|>
Device: {device_type}
Visual: {visual_clean}
Internal: {issues_clean}

Response:
Diagnosis:
<|assistant|>
"""

        # 3. Generate
        # ctransformers returns the string directly! No complex decoding needed.
        generated_text = self.llm(
            prompt,
            max_new_tokens=250,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.2,
            stop=["</s>", "<|user|>"] # Stop if it tries to generate a user turn
        )

        # 4. Cleanup & Format
        final_text = "Diagnosis: " + generated_text if not generated_text.strip().startswith("Diagnosis:") else generated_text
        return self._format_output(final_text)

    def _format_output(self, text):
        # (Keep this helper method exactly the same as you have it now)
        lines = text.split('\n')
        result = {}
        current_key = "Summary"
        target_keys = ["Diagnosis", "Severity", "Action", "Reasoning"]
        
        for line in lines:
            found_key = False
            for key in target_keys:
                if line.strip().startswith(key + ":"):
                    _, val = line.split(":", 1)
                    result[key] = val.strip()
                    current_key = key
                    found_key = True
                    break
            
            if not found_key and current_key in result and line.strip():
                result[current_key] += " " + line.strip()
        
        final_str = ""
        for k in target_keys:
            if k in result:
                final_str += f"{k}: {result[k]}\n"
        
        return final_str if final_str.strip() else text

# --- MAIN PIPELINE CLASS ---
class DiagnosisPipeline:
    # In backend.py inside DiagnosisPipeline class

    def __init__(self):
        self.config = Config()
        
        # --- MEMORY CLEANUP ---
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # --- 1. LOAD MODELS ---
        self._load_llm()        # This is fast now (GGUF)
        self._load_yolo()       # This is local (Fast)
        self._load_classifier() # This is local (Fast)
        
        # --- CHANGE THIS SECTION ---
        # self._load_nlp()  <-- DELETE THIS LINE (It causes the timeout)
        
        # Add these lines instead to act as placeholders
        self.nlp_model = None
        self.nlp_tokenizer = None

    def _load_llm(self):
        # We pass the token (if needed) and model ID
        self.advisor = DeviceAdvisorLLM(self.config.LLM_MODEL_ID, self.config.HF_TOKEN)

    def _load_yolo(self):
        try:
            self.yolo = YOLO(self.config.YOLO_PATH)
        except:
            logger.warning("Using fallback YOLOv8n")
            self.yolo = YOLO("yolov8n.pt")

    def _load_classifier(self):
        # RESNET50
        self.classifier = timm.create_model('resnet50', pretrained=False, num_classes=9)
        try:
            state_dict = torch.load(self.config.CLASSIFIER_PATH, map_location=self.config.DEVICE)
            self.classifier.load_state_dict(state_dict)
            self.classifier.to(self.config.DEVICE).eval()
        except:
            logger.warning("Visual Classifier weights not found. Visual checks will be skipped.")
            self.classifier = None
        
        self.img_transforms = transforms.Compose([
            transforms.Resize((512, 512)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_nlp(self):
        try:
           
            self.nlp_tokenizer = DistilBertTokenizer.from_pretrained(
                self.config.NLP_PATH,
                subfolder="electronics_nlp_model" 
            )
            self.nlp_model = DistilBertForSequenceClassification.from_pretrained(
                self.config.NLP_PATH,
                subfolder="electronics_nlp_model"  
            )
            self.nlp_model.to(self.config.DEVICE).eval()
            logger.info(f"NLP Model loaded from {self.config.NLP_PATH}")
        except Exception as e:
            logger.error(f"NLP Model FAILED to load: {e}")

    def analyze_case(self, image_path, user_comment):
        logger.info(f"Analyzing: {image_path}")
        
        # Defaults
        device_type = "Unknown Device"
        visual_condition = "N/A" # Default to N/A
        
        try:
            img = Image.open(image_path).convert("RGB")
            results = self.yolo(img)
            
            detected_box = None
            
            # 1. YOLO DETECTION
            for box in results[0].boxes:
                cls_id = int(box.cls)
                device_type = self.yolo.names[cls_id] # Update device type
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_box = (x1, y1, x2, y2)
                break 
            
            # List of devices we have a ResNet for
            screen_devices = [
                'Laptop', 'Smartphone', 'Flat-Panel-Monitor', 'Flat-Panel-TV', 
                'Desktop-PC', 'Digital-Oscilloscope', 'Telephone-Set'
            ]
            
            # 2. ROUTING LOGIC
            if detected_box:
                # CASE A: It is a screen device -> Use ResNet
                if device_type in screen_devices and self.classifier:
                    crop = img.crop(detected_box)
                    visual_condition = self._classify_crop(crop)
                
                # CASE B: Known device, but not a screen (e.g., Drone) -> Skip ResNet
                else:
                    visual_condition = "N/A (Non-screen device)"
            
            # CASE C: YOLO saw nothing
            elif not detected_box:
                logger.warning("YOLO: No object detected.")
                device_type = "Unknown (Rely on Text)"
                visual_condition = "N/A"

        except Exception as e:
            logger.error(f"Visual pipeline error: {e}")

        # 3. NLP & LLM
        nlp_issues = self._analyze_text(user_comment)

        recommendation = self.advisor.generate_recommendation(
            device_type=device_type,
            visual_condition=visual_condition,
            nlp_issues=nlp_issues
        )

        return {
            "device_detected": device_type,
            "visual_condition": visual_condition,
            "nlp_issues": nlp_issues,
            "recommendation": recommendation
        }

    def _classify_crop(self, pil_img):
        # (Your existing logic)
        tensor = self.img_transforms(pil_img).unsqueeze(0).to(self.config.DEVICE)
        with torch.no_grad():
            output = self.classifier(tensor)
            probs = F.softmax(output[0], dim=0)
            top_prob, top_idx = torch.max(probs, 0)
            if top_prob.item() < 0.4:
                return "Uncertain/Normal"
            return self.config.IMG_CLASSES[top_idx.item()]

    def _analyze_text(self, text):
        # 1. Lazy Load: If models aren't loaded, try loading them now
        if self.nlp_model is None:
            logger.info("Lazy loading NLP model...")
            self._load_nlp()

        # 2. CRITICAL SAFETY CHECK
        # If _load_nlp failed (e.g. bad path), these will still be None.
        # We must return an empty list immediately to prevent the crash.
        if self.nlp_model is None or self.nlp_tokenizer is None:
            return []
            
        # 3. Run Analysis (Only if we passed the check above)
        try:
            inputs = self.nlp_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.config.DEVICE)
            with torch.no_grad():
                outputs = self.nlp_model(**inputs)
                probs = torch.sigmoid(outputs.logits).squeeze()
            
            detected = []
            for i, prob in enumerate(probs):
                if prob > 0.5:
                    detected.append(self.config.NLP_LABELS[i])
            return detected
        except Exception as e:
            logger.error(f"Error during NLP analysis: {e}")
            return []

    def _classify_crop(self, pil_img):
        tensor = self.img_transforms(pil_img).unsqueeze(0).to(self.config.DEVICE)
        with torch.no_grad():
            output = self.classifier(tensor)
            probs = F.softmax(output[0], dim=0)
            top_prob, top_idx = torch.max(probs, 0)
            
            if top_prob.item() < 0.4:
                return "Uncertain/Normal"
            return self.config.IMG_CLASSES[top_idx.item()]

    def _analyze_text(self, text):
        if not hasattr(self, 'nlp_model'): return []
        inputs = self.nlp_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.config.DEVICE)
        with torch.no_grad():
            outputs = self.nlp_model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze()
        
        detected = []
        for i, prob in enumerate(probs):
            if prob > 0.5:
                detected.append(self.config.NLP_LABELS[i])
        return detected