# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/ruT5-large")