import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    DistilBertTokenizer, 
    DistilBertForSequenceClassification
)
from ultralytics import YOLO
import timm
from torchvision import transforms
import logging
import gc

class Config:
    YOLO_PATH = "electronics_type_classifier/runs/detect/train4/weights/best.pt"
    CLASSIFIER_PATH = "condition_classifier/defect_classifier_v1.pth"
    NLP_PATH = "keyword_extraction/electronics_nlp_model"
    
    LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    IMG_CLASSES = [
        'laptop_normal', 'screen_crack', 'laptop_fades', 'laptop_lines', 
        'laptop_spot', 'phone_dead_pixel', 'phone_scratch', 'screen_crack', 'phone_normal'
    ]
    
    NLP_LABELS = [
        'Power_Failure', 'Battery_Charging', 'Display_Visual', 'Audio_Sound',
        'Overheating_Thermal', 'Connectivity_Signal', 'Water_Liquid_Damage',
        'Mechanical_Motor', 'Input_Controls', 'Software_Error', 'Data_Storage',
        'Sensor_Accuracy'
    ]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EcoScan")

class DeviceAdvisorLLM:
    def __init__(self, model_id):
        logger.info(f"Loading LLM: {model_id}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def generate_recommendation(self, device_type, visual_condition, nlp_issues, raw_text):
        visual_clean = visual_condition.replace("_", " ").title()
        issues_clean = ", ".join([x.replace("_", " ") for x in nlp_issues]) if nlp_issues else "None"

        messages = [
            {"role": "system", "content": (
                "You are a technical diagnostic tool for e-waste. "
                "Output the status in a specific format. Keep reasoning brief."
            )},
            {"role": "user", "content": (
                f"Device: {device_type}\n"
                f"Visual Defect: {visual_clean}\n"
                f"Reported Issues: {issues_clean}\n"
                f"User Complains: {raw_text}\n\n"
                "Provide a diagnosis with Severity (Low/Medium/High), Recommended Action, and short Reasoning."
            )}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class DiagnosisPipeline:
    def __init__(self):
        self.config = Config()
        self._cleanup_memory()
        
        self._load_llm()
        self._load_yolo()
        self._load_classifier()
        self._load_nlp()

    def _cleanup_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _load_llm(self):
        self.advisor = DeviceAdvisorLLM(self.config.LLM_MODEL_ID)

    def _load_yolo(self):
        try:
            self.yolo = YOLO(self.config.YOLO_PATH)
        except Exception:
            logger.warning("Custom YOLO not found, using fallback.")
            self.yolo = YOLO("yolov8n.pt")

    def _load_classifier(self):
        self.classifier = timm.create_model('resnet50', pretrained=False, num_classes=9)
        try:
            state_dict = torch.load(self.config.CLASSIFIER_PATH, map_location=self.config.DEVICE)
            self.classifier.load_state_dict(state_dict)
            self.classifier.to(self.config.DEVICE).eval()
        except:
            logger.warning("Visual Classifier weights not found.")
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

    def _get_visual_prediction(self, img_tensor):
        """Helper to run ResNet inference on a tensor"""
        if self.classifier is None: return "N/A", 0.0
        
        with torch.no_grad():
            output = self.classifier(img_tensor)
            probs = F.softmax(output[0], dim=0)
            top_prob, top_idx = torch.max(probs, 0)
            
            label = self.config.IMG_CLASSES[top_idx.item()]
            return label, top_prob.item()

    def analyze_case(self, image_path, user_comment):
        logger.info(f"Analyzing: {image_path}")
        
        original_img = Image.open(image_path).convert("RGB")
        nlp_issues = self._analyze_text(user_comment)
        
        
        yolo_results = self.yolo(original_img, verbose=False)
        yolo_box = None
        yolo_class = None
        yolo_conf = 0.0
        
        for box in yolo_results[0].boxes:
            yolo_conf = float(box.conf)
            if yolo_conf > 0.7: 
                cls_id = int(box.cls)
                yolo_class = self.yolo.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_box = (x1, y1, x2, y2)
                break
        
        resnet_label = "Uncertain"
        resnet_conf = 0.0
        
        if yolo_box:
            logger.info(f"YOLO Detected: {yolo_class} ({yolo_conf:.2f}). Cropping...")
            crop = original_img.crop(yolo_box)
            tensor = self.img_transforms(crop).unsqueeze(0).to(self.config.DEVICE)
            resnet_label, resnet_conf = self._get_visual_prediction(tensor)
        else:
            logger.info("YOLO Failed. Attempting Full Image Classification...")
            tensor = self.img_transforms(original_img).unsqueeze(0).to(self.config.DEVICE)
            resnet_label, resnet_conf = self._get_visual_prediction(tensor)

        
        final_device = "Unknown Device"
        final_condition = "N/A"
        
        
        if resnet_conf > 0.8 and yolo_conf < 0.8:
            final_condition = resnet_label
            if "_" in resnet_label:
                inferred_device = resnet_label.split("_")[0].capitalize()
                final_device = inferred_device
                logger.info(f"ResNet Override! Inferred {final_device} from defect analysis.")
                
        elif yolo_class:
            final_device = yolo_class
            final_condition = resnet_label if resnet_conf > 0.4 else "Unknown Condition"
            
        else:
            final_device = "Unknown"
            final_condition = "Unrecognizable"

        recommendation = self.advisor.generate_recommendation(
            device_type=final_device,
            visual_condition=final_condition,
            nlp_issues=nlp_issues,
            raw_text=user_comment
        )

        return {
            "device_detected": final_device,
            "visual_condition": final_condition,
            "confidence_scores": {"yolo": yolo_conf, "resnet": resnet_conf},
            "nlp_issues": nlp_issues,
            "recommendation": recommendation,
            "raw_text": user_comment
        }

    def _analyze_text(self, text):
        if not hasattr(self, 'nlp_model'): return []
        inputs = self.nlp_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.config.DEVICE)
        with torch.no_grad():
            outputs = self.nlp_model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze()
        
        detected = []
        if probs.dim() == 0: probs = probs.unsqueeze(0)
        for i, prob in enumerate(probs):
            if prob > 0.5:
                detected.append(self.config.NLP_LABELS[i])
        return detected