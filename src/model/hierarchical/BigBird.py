from typing import Any
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from src.model.AbstractModel import AbstractModel
from src.model.add_special_tokens_and_initialize import add_special_tokens_and_initialize


class BigBird(AbstractModel):
    def __init__(
        self, 
        pretrained_model_name_or_path,
        additional_special_tokens: list = ['<person>'], 
        personal_pronouns: list = ['君', 'きみ', 'あなた' ,'彼', '彼女'], 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
    
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.fn = nn.Linear(self.model.config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Resize for the <person> token
        if additional_special_tokens is not None and '<person>' in additional_special_tokens and len(additional_special_tokens) == 1:
            add_special_tokens_and_initialize(self.model, pretrained_model_name_or_path, additional_special_tokens, personal_pronouns)
        
    def forward(
        self, 
        input_ids, 
        attention_mask,
        output_attentions=False,
        labels=None,
    ):
        try:
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_attentions=output_attentions
            )
            preds = self.fn(outputs.pooler_output)
        except RuntimeError as e:
            self.logger.error(f"RuntimeError has occurred. {e}")
            raise e
        loss = None
        if labels is not None:
            loss = self.criterion(preds, labels)
        
        # release gpu memory
        preds_detached = preds.clone().detach()
        attentions_detached = outputs.attentions.clone().detach() if outputs.attentions is not None else None
        del outputs, preds
        torch.cuda.empty_cache()
        
        return dict(loss=loss, preds=preds_detached, attentions=attentions_detached)


    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 100 == 0:
            preds = torch.argmax(outputs['preds'], dim=1)
            for i in range(len(batch['input_ids'])):
                result = "正解" if preds[i] == batch['labels'][i] else "不正解"
                sample_text = self.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=False)
                self.logger.experiment.add_text(f"{result}(label:{batch['labels'][i]},pred{preds[i]})", f"{sample_text}", self.current_epoch)

        return dict(loss=outputs['loss'], batch_preds=outputs['preds'], batch_labels=batch['labels'])
    
    
    def get_optimizer_params(self, lr, weight_decay=0.0):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters
    
    
    def configure_optimizers(self):
        optimizer_parameters = self.get_optimizer_params(self.hparams.optim.args.lr, self.hparams.optim.args.weight_decay)
        optimizer = torch.optim.AdamW(optimizer_parameters)
        return optimizer
    