import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Convert Q/A pairs to fine-tuning format (JSONL)
def convert_qa_to_jsonl(input_json='qa_pairs.json', output_jsonl='qa_pairs_ft.json'):
    with open(input_json, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for pair in qa_pairs:
            record = {
                'instruction': pair['Q'],
                'output': pair['A']
            }
            f_out.write(json.dumps(record) + '\n')
    print(f"Saved fine-tuning dataset to {output_jsonl}")

convert_qa_to_jsonl()

# Load dataset
dataset = load_dataset('json', data_files={'train': 'qa_pairs_ft.json'})
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

# Preprocess function
def preprocess_function(example):
    input_text = example['instruction'] + '\nAnswer:'
    target_text = example['output']
    model_inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=128)
    labels = tokenizer(target_text, padding='max_length', truncation=True, max_length=64)['input_ids']
    model_inputs['labels'] = labels
    return model_inputs

train_dataset = dataset['train'].map(preprocess_function, remove_columns=['instruction', 'output'])

# Load model
model = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=None)

# Training arguments
training_args = TrainingArguments(
    output_dir='./ft_results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_dir='./ft_logs',
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to=['none'],
    remove_unused_columns=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Run fine-tuning
trainer.train()
# Save the fine-tuned model
trainer.save_model('ft_distilgpt2_model')
print("Fine-tuning complete. Model saved to 'ft_distilgpt2_model'.")
