import torch
from transformers import AutoTokenizer, BertForSequenceClassification

class EmotionClassifier:
    def __init__(self, model_path="./emotion_classifier_final", threshold=0.3, top_k=2):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.threshold = threshold
        self.top_k = top_k

        # GoEmotions label list
        self.labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                       'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                       'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                       'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                       'relief', 'remorse', 'sadness', 'surprise', 'neutral']

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0]

        # Emotions above threshold
        high_conf = [(label, p.item()) for label, p in zip(self.labels, probs) if p > self.threshold]

        if not high_conf:
            # Get top k emotions when none exceed threshold
            top_indices = torch.topk(probs, k=self.top_k).indices
            high_conf = [(self.labels[idx], probs[idx].item()) for idx in top_indices]

        return {
            "text": text,
            "predictions": sorted(high_conf, key=lambda x: x[1], reverse=True)
        }

    def print_prediction(self, text):
        result = self.predict(text)
        print(f"\nText: {result['text']}\nPredicted emotions:")
        for label, prob in result["predictions"]:
            print(f"  {label}: {prob:.2f}")
