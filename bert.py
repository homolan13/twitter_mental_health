# Some parts copied from https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import argparse

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Necessary pre-processing
def preprocessing(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
        - input_ids: list of token ids
        - token_type_ids: list of token type ids
        - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens=True,
                        max_length=64,
                        truncation=True,
                        padding='max_length',
                        return_attention_mask=True,
                        return_tensors='pt'
                )

def metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds) # aka sensitivity
    f1 = f1_score(labels, preds)

    return accuracy, precision, recall, f1


def main(batch_size, n_epochs, max_patience):
    raw = pd.read_csv('./data/Mental-Health-Twitter.csv', index_col=0)
    data = raw[['post_text', 'label']]

    # Load BERT tokenizer and classifier as well as the optimizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
    model.cuda()
    optimizer = th.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-08)

    # Pre-process data
    token_id = []
    attention_masks = []
    for tweet in data['post_text'].values:
        encoding_dict = preprocessing(tweet, tokenizer)
        token_id.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])

    token_id = th.cat(token_id, dim = 0)
    attention_masks = th.cat(attention_masks, dim = 0)
    labels = th.tensor(data['label'].values)

    # Create train, validation and test sets and prepare DataLoader
    train_val_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.3, shuffle=True, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, shuffle=True, stratify=labels[train_val_idx])
    train_set = TensorDataset(token_id[train_idx], attention_masks[train_idx], labels[train_idx])
    train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
    val_set = TensorDataset(token_id[val_idx], attention_masks[val_idx], labels[val_idx])
    val_dataloader = DataLoader(val_set, sampler=RandomSampler(val_set), batch_size=batch_size)
    test_set = TensorDataset(token_id[test_idx], attention_masks[test_idx], labels[test_idx])
    test_dataloader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=batch_size)

    # Prepare training and validation/test output
    patience = max_patience
    best_accuracy = 0 # Keep track of best performance for patience
    it = 0
    val_output = {k: [] for k in ['training_loss', 'accuracy', 'precision', 'recall', 'f1']}
    test_output = {k: 'NaN' for k in ['accuracy', 'precision', 'recall', 'f1']}

    for epoch in range(n_epochs):
        
        ### TRAINING ###
        # Set model to training mode
        model.train()

        for batch in tqdm(train_dataloader):
            it += 1
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass
            train_output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # Backward pass
            train_output.loss.backward()
            optimizer.step()
            # Log loss every 20th iteration
            if it%20 == 0:
                val_output['training_loss'].append((it, train_output.loss.item()))

            # Log metrics every 100th iteration
            if it%100 == 0:
                # Set model to evaluation mode
                model.eval()

                # Tracking variables 
                batch_accuracy = []
                batch_precision = []
                batch_recall = []
                batch_f1 = []

                for batch in val_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch
                    with th.no_grad():
                        # Forward pass
                        eval_output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    predictions = eval_output.logits.detach().cpu().numpy()
                    predictions = np.argmax(predictions, axis=1).flatten()
                    label_ids = b_labels.to('cpu').numpy()
                    # Calculate validation metrics
                    b_accuracy, b_precision, b_recall, b_f1 = metrics(label_ids, predictions)
                    batch_accuracy.append(b_accuracy)
                    batch_precision.append(b_precision)
                    batch_recall.append(b_recall)
                    batch_f1.append(b_f1)

                val_output['accuracy'].append((it, np.array(batch_accuracy).mean()))
                val_output['precision'].append((it, np.array(batch_precision).mean()))
                val_output['recall'].append((it, np.array(batch_recall).mean()))
                val_output['f1'].append((it, np.array(batch_f1).mean()))
                
                print(f'Loss: {val_output["training_loss"][-1][1]:.3f}')
                print(f'Accuracy: {val_output["accuracy"][-1][1]:.3f}')

                # Early stopping
                if val_output['accuracy'][-1][1] > best_accuracy:
                    best_accuracy = val_output['accuracy'][-1][1]
                    best_model_state = model.state_dict()
                    th.save(best_model_state, './output/bert_model_ft.pt')
                    patience = max_patience
                    print(f'Resetting patience to {max_patience}')
                else:
                    patience -= 1
                    print(f'Patience: {patience}')

                if patience == 0:
                    print(f'Patience run out.')
                    break

        else: # If patience is not run out, continue with training
            continue
        break

    val_output['n_epochs'] = epoch

    ### TESTING ###    
    # Set best model to evaluation mode
    best_model = model
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    # Tracking variables 
    batch_accuracy = []
    batch_precision = []
    batch_recall = []
    batch_f1 = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with th.no_grad():
            # Forward pass
            eval_output = best_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        predictions = eval_output.logits.detach().cpu().numpy()
        predictions = np.argmax(predictions, axis=1).flatten()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate validation metrics
        b_accuracy, b_precision, b_recall, b_f1 = metrics(label_ids, predictions)
        batch_accuracy.append(b_accuracy)
        batch_precision.append(b_precision)
        batch_recall.append(b_recall)
        batch_f1.append(b_f1)

    test_output['accuracy'] = np.array(batch_accuracy).mean()
    test_output['precision'] = np.array(batch_precision).mean()
    test_output['recall'] = np.array(batch_recall).mean()
    test_output['f1'] = np.array(batch_f1).mean()

    print('\n\nTest results:')
    print(f'Accuracy: {test_output["accuracy"]:.3f}')
    print(f'Precision: {test_output["precision"]:.3f}')
    print(f'Recall: {test_output["recall"]:.3f}')
    print(f'F1: {test_output["f1"]:.3f}')

    # Save output
    with open('./output/bert_output_train.json', 'w') as f:
        json.dump(val_output, f)
    with open('./output/bert_output_test.json', 'w') as f:
        json.dump(test_output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=6)
    parser.add_argument('--max_patience', type=int, default=8)
    args = parser.parse_args()
    main(batch_size=args.batch_size, n_epochs=args.n_epochs, max_patience=args.max_patience)