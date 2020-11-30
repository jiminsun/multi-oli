import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TSNE:
    def __init__(self, model, dataset, device, exp_name):
        self.model = model.to(device)
        self._id_to_lang = dataset._id_to_lang
        self.dataloader = DataLoader(dataset,
                                     batch_size=500,
                                     num_workers=5, shuffle=False)
        self.device = device
        self.writer = SummaryWriter(exp_name)

    def visualize(self):
        self.dataloader.repeat = False
        embeddings = []
        language = []
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                print(f'Processing batch {idx}')
                outputs = self.model.model.bert(batch['input_ids'].to(self.device),
                                                batch['attn_mask'].to(self.device))
                last_hidden_state = outputs[0]
                emb = self.model.model.pool(last_hidden_state=last_hidden_state,  # [B x E]
                                            attention_mask=batch['attn_mask'].to(self.device))
                embeddings.append(emb)
                language.append(batch['lang'])
        embeddings = torch.cat(embeddings, dim=0)
        language = torch.cat(language, dim=0).cpu().tolist()
        category = [self._id_to_lang[idx] for idx in language]
        self.writer.add_embedding(
            embeddings,
            metadata=category
        )
        self.writer.close()
