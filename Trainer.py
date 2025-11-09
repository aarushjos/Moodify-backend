import re
import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer,default_data_collator

print(f"Cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Gpu: {torch.cuda.get_device_name(0)}")
else:
    print("Cuda not available")


dataset=load_dataset("go_emotions")
print(dataset["train"][0])
labels=dataset["train"].features["labels"].feature.names
print(labels)

#preprocessing the text
def clean_text(example):
    text=example['text'].lower()
    text=re.sub(r"http\S+",",",text) #replace urls with ""
    text=re.sub(r"[^a-zA-Z\s]","",text) #replace punctuation with ""
    text=re.sub(r"\s+"," ",text).strip() #replace multiple space with one "     " -> " "
    example["text"]=text
    return example

dataset=dataset.map(clean_text,num_proc=1) #applies fun to each example in dataset


tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset=dataset.map(tokenize,batched=True)


#each example may have multiple emotion so we convert to one-hot vector

def encode_labels(example):
    # PyTorch needs FloatTensors and not LongTensors
    label_vec=np.zeros(len(labels),dtype=np.float32)
    label_vec[example["labels"]]=1.0
    example["labels"]=label_vec.tolist()
    return example

encoded_dataset=tokenized_dataset.map(encode_labels)


encoded_dataset=encoded_dataset.remove_columns(["text"])
encoded_dataset=encoded_dataset.remove_columns(["id"])

encoded_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
)

def custom_collator(features):
    batch = default_data_collator(features)
    # Ensure labels are float32
    batch["labels"] = batch["labels"].float()
    return batch


print(encoded_dataset["train"][0]["labels"].dtype)
print(encoded_dataset["train"][0]) #labels-> multi-hot encoded vector, id-> refers to id of text,
# input_ids-> tokenized text in numeric form for model [101->start of sentence,102->end], token_type_ids-> distinguishes sentence A with sentence B but 0 here cause no sentence pair in dataset,
# attention_mask-> which token to use during training and which to ignore (1->token,2->padding)

model=BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=len(labels),problem_type="multi_label_classification")


training_args=TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    fp16=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=0,
    gradient_accumulation_steps=1,
)


def metrics(prediction):
    logits,labels=prediction
    probs=torch.sigmoid(torch.tensor(logits))  #sigmoid=1/1+e^-z
    y_pred=(probs>0.5).int()
    y_true=torch.tensor(labels)
    f1=f1_score(y_true, y_pred, average="micro")
    return{"f1":f1}

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=metrics,
    data_collator=custom_collator,
)


trainer.train()

trainer.save_model("./emotion_classifier_final")
results=trainer.evaluate(encoded_dataset["test"])
print(f"Results: {results}")