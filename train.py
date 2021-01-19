import sys
import getopt

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

from utils import load_dataset, load_data_collator

def train(
    train_file_path,
    model_name_or_type='gpt2',

    output_dir='./results',
    overwrite_output_dir=False,
    per_device_train_batch_size=8,
    num_train_epochs=3.0,
    logging_dir='./logs',
    loggint_steps=10
    ):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_type)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        logging_dir=logging_dir,
        logging_steps=loggint_steps
    )

    model = GPT2LMHeadModel.from_pretrained(model_name_or_type)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:],
        '',
        [
            'train_file_path=',
            'model_name_or_type=',
            'output_dir=',
            'overwrite_output_dir',
            'per_device_train_batch_size=',
            'num_train_epochs=',
            'logging_dir=',
            'logging_steps=',
        ])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    
    train_file_path = None
    model_name_or_type='gpt2'

    output_dir='./results'
    overwrite_output_dir=False
    per_device_train_batch_size=8
    num_train_epochs=3.0
    logging_dir='./logs'
    logging_steps=10

    for o, a in opts:
        if o == '--train_file_path':
            train_file_path = a
        elif o == '--model_name_or_type':
            model_name_or_type = a
        elif o == '--output_dir':
            output_dir = a
        elif o == '--overwrite_output_dir':
            overwrite_output_dir = True
        elif o == '--per_device_train_batch_size':
            per_device_train_batch_size = int(a)
        elif o == '--num_train_epochs':
            num_train_epochs = float(a)
        elif o == '--logging_dir':
            logging_dir = a
        elif o == '--logging_steps':
            logging_steps = int(a)
    if train_file_path is None:
        sys.exit(3)

    train(
        train_file_path=train_file_path,
        model_name_or_type=model_name_or_type,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        logging_dir=logging_dir,
        loggint_steps=logging_steps
    )

if __name__ == '__main__':
    main()