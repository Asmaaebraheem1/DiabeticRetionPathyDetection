from transformers import AutoModelForImageClassification, AutoImageProcessor

model_name = "Asmaa111/diabetic-eye"
save_path = "./diabetic_model"

model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

model.save_pretrained(save_path)
processor.save_pretrained(save_path)
