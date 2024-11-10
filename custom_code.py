import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Load the model and tokenizer
model_name = 'gpt2'  # Replace with your model of choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Define the dataset
class ORPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]['prompt']
        chosen = self.data[idx]['chosen']
        rejected = self.data[idx]['rejected']

        # Tokenize prompt + chosen
        chosen_encodings = self.tokenizer(
            prompt + chosen,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length'
        )

        # Tokenize prompt + rejected
        rejected_encodings = self.tokenizer(
            prompt + rejected,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length'
        )

        # Tokenize prompt separately to identify its length
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        prompt_length = prompt_encodings['input_ids'].size(1)

        return {
            'input_ids': torch.cat([chosen_encodings['input_ids'], rejected_encodings['input_ids']], dim=0).squeeze(1),
            'attention_mask': torch.cat([chosen_encodings['attention_mask'], rejected_encodings['attention_mask']], dim=0).squeeze(1),
            'labels': torch.cat([chosen_encodings['input_ids'], rejected_encodings['input_ids']], dim=0).squeeze(1),
            'prompt_length': prompt_length
        }

# Example data
data = [
    {
        'prompt': 'Once upon a time',
        'chosen': ' there was a brave knight who fought dragons.',
        'rejected': ' there was a big dragon that ate people.'
    },
    # Add more data samples here
]

# Create the dataset and dataloader
dataset = ORPODataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Define the ORPO loss function
def orpo_loss(chosen_logps, rejected_logps, beta=0.1):
    # Compute the odds ratio loss
    log_odds = (chosen_logps - rejected_logps) - (
        torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
    )
    ratio = F.logsigmoid(log_odds)
    losses = beta * ratio
    # Return the mean loss
    return -losses.mean()

# Function to compute log probabilities from logits
def compute_sequence_log_probs_from_logits(logits, labels, attention_mask):
    # Shift logits and labels for causal language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probabilities at the positions of the input tokens
    selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # Apply attention mask
    selected_log_probs = selected_log_probs * shift_attention_mask

    # Sum the log probabilities over the sequence length
    total_log_probs = selected_log_probs.sum(dim=1)

    # Compute average log probability per token
    seq_lengths = shift_attention_mask.sum(dim=1)
    avg_log_probs = total_log_probs / seq_lengths

    return avg_log_probs

# Training loop
model.train()
num_epochs = 3

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Move tensors to the appropriate device if using GPU
        # batch = {k: v.to(device) for k, v in batch.items()}

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        prompt_length = batch['prompt_length']

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        nll_loss = outputs.loss  # This is the CrossEntropyLoss over the entire output

        # Split logits back to chosen and rejected
        batch_size = input_ids.size(0) // 2
        chosen_logits = logits[:batch_size]
        rejected_logits = logits[batch_size:]
        chosen_input_ids = input_ids[:batch_size]
        rejected_input_ids = input_ids[batch_size:]
        chosen_attention_mask = attention_mask[:batch_size]
        rejected_attention_mask = attention_mask[batch_size:]
        chosen_labels = labels[:batch_size]
        rejected_labels = labels[batch_size:]

        # Compute log probabilities for ORPO loss
        chosen_log_probs = compute_sequence_log_probs_from_logits(
            chosen_logits,
            chosen_labels,
            chosen_attention_mask
        )
        rejected_log_probs = compute_sequence_log_probs_from_logits(
            rejected_logits,
            rejected_labels,
            rejected_attention_mask
        )

        # Compute ORPO loss
        orpo_loss_value = orpo_loss(chosen_log_probs, rejected_log_probs, beta=0.1)

        # Total loss
        loss = nll_loss - orpo_loss_value

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, NLL Loss: {nll_loss.item():.4f}, ORPO Loss: {orpo_loss_value.item():.4f}")
