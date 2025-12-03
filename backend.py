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

class Config:
    # Model Paths
    YOLO_PATH = "electronics_type_classifier/runs/detect/train4/weights/best.pt"
    CLASSIFIER_PATH = "condition_classifier/defect_classifier_v1.pth"
    NLP_PATH = "keyword_extraction/electronics_nlp_model"
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
        logger.info(f"Loading LLM: {model_id}...")
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.float32,  # Use standard precision for CPU
            device_map="cpu",           # Force CPU loading
            low_cpu_mem_usage=True
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15
        )

    def generate_recommendation(self, device_type, visual_condition, nlp_issues):
        visual_clean = visual_condition.replace("_", " ").title()
        issues_clean = ", ".join([x.replace("_", " ") for x in nlp_issues]) if nlp_issues else "None"

        # 1. Prompt with negative constraints and one-shot example
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
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # 2. Generation with higher token limit (250)
        outputs = self.model.generate(
            inputs, 
            max_new_tokens=250,       
            temperature=0.2,          
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 3. Cleanup logic
        if "<|assistant|>" in raw_response:
            generated_text = raw_response.split("<|assistant|>")[-1].strip()
        else:
            generated_text = raw_response

        # Force "Diagnosis:" prefix if missing (since we forced it in prompt)
        final_text = "Diagnosis: " + generated_text if not generated_text.startswith("Diagnosis:") else generated_text
            
        return self._format_output(final_text)

    def _format_output(self, text):
        """
        Parses the raw text into a clean string even if the model messes up slightly.
        """
        # Fallback parsing strategy: split by newlines and reconstruct
        lines = text.split('\n')
        result = {}
        current_key = "Summary"
        
        # Keys we expect to find
        target_keys = ["Diagnosis", "Severity", "Action", "Reasoning"]
        
        for line in lines:
            # Check if this line starts with one of our keys
            found_key = False
            for key in target_keys:
                if line.strip().startswith(key + ":"):
                    _, val = line.split(":", 1)
                    result[key] = val.strip()
                    current_key = key
                    found_key = True
                    break
            
            # If it's a continuation of the previous line (model rambling)
            if not found_key and current_key in result and line.strip():
                result[current_key] += " " + line.strip()
        
        # Rebuild the final string to look clean
        final_str = ""
        for k in target_keys:
            if k in result:
                final_str += f"{k}: {result[k]}\n"
        
        # If regex failed completely, just return the raw text so we don't return empty string
        return final_str if final_str.strip() else text

# --- MAIN PIPELINE CLASS ---
class DiagnosisPipeline:
    def __init__(self):
        self.config = Config()
        
        # --- MEMORY CLEANUP ---
        # Clear any leftover garbage from previous runs
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- 1. LOAD THE LLM FIRST ---
        # The LLM is the "biggest rock". We load it first to ensure 
        # it gets the continuous VRAM block it needs for 4-bit quantization.
        self._load_llm()
        
        # --- 2. LOAD SMALLER MODELS ---
        # These fit in the gaps or can be managed more easily by PyTorch
        self._load_yolo()
        self._load_classifier()
        self._load_nlp()

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
            self.nlp_tokenizer = DistilBertTokenizer.from_pretrained(self.config.NLP_PATH)
            self.nlp_model = DistilBertForSequenceClassification.from_pretrained(self.config.NLP_PATH)
            self.nlp_model.to(self.config.DEVICE).eval()
        except:
            logger.warning("NLP Model not found.")

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
        # (Your existing logic)
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