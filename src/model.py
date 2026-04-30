from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch
import ot
import numpy as np
import torch.nn.functional as F
from geomloss import SamplesLoss
from sklearn.manifold import MDS
import geomloss
import pysdtw
from pysdtw.distance import pairwise_l2_squared



class ModelWithEMDLoss(nn.Module):
    def __init__(self, base_model_name, token_ids_dict, tokenizer,lora_config=None, margin=0.3, cfg=None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

        if "Mistral" in base_model_name:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id



        if cfg is not None:
            self.cfg = cfg
 
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)
        self.tokenizer = tokenizer
        self.token_ids_dict = token_ids_dict
        self.margin = margin

        self.duration_embeddings = nn.Parameter(torch.randn(9, 4))
        self.OTLoss=SamplesLoss(
            loss='sinkhorn',
            p=2,
            blur=0.1 ** (1 / 2),
            backend='tensorized'
        )

        self.OTLoss_cost = SamplesLoss(
            loss='sinkhorn',
            p=2,
            cost=self.cost,
            blur=0.1 ** (1 / 2),
            backend='tensorized'
        )

        self.cost_matrix = torch.tensor(np.array([
            [[0.000, 4.094, 8.188, 12.282, 13.140, 14.013, 17.269, 19.469, 21.669],
             [4.094, 0.000, 4.094, 8.188, 9.046, 9.919, 13.175, 15.375, 17.575],
             [8.188, 4.094, 0.000, 4.094, 4.952, 5.825, 9.081, 11.281, 13.481],
             [12.282, 8.188, 4.094, 0.000, 0.858, 1.731, 4.987, 7.187, 9.387],
             [13.140, 9.046, 4.952, 0.858, 0.000, 0.873, 4.129, 6.329, 8.529],
             [14.013, 9.919, 5.825, 1.731, 0.873, 0.000, 3.256, 5.456, 7.656],
             [17.269, 13.175, 9.081, 4.987, 4.129, 3.256, 0.000, 2.200, 4.400],
             [19.469, 15.375, 11.281, 7.187, 6.329, 5.456, 2.200, 0.000, 2.200],
             [21.669, 17.575, 13.481, 9.387, 8.529, 7.656, 4.400, 2.200, 0.000]]
            ]), dtype=torch.float32)

     
    def forward(self, input_ids, attention_mask=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output.logits
        last_token_index = attention_mask.sum(dim=1) - 1  # [B]
        batch_size = logits.size(0)
        last_token_logits = logits[torch.arange(batch_size), last_token_index, :]  # [B, V]


        return last_token_logits  # [B, num_bins]

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

  

    def cost(self, x, y):
        cost_matrix = torch.tensor(np.array([
            [0.000, 1.778, 3.556, 4.934, 5.781, 6.414, 7.498, 8.498, 9.498],
            [1.778, 0.000, 1.778, 3.156, 4.003, 4.636, 5.720, 6.720, 7.720],
            [3.556, 1.778, 0.000, 1.378, 2.225, 2.858, 3.942, 4.942, 5.942],
            [4.934, 3.156, 1.378, 0.000, 0.847, 1.480, 2.564, 3.564, 4.564],
            [5.781, 4.003, 2.225, 0.847, 0.000, 0.633, 1.717, 2.717, 3.717],
            [6.414, 4.636, 2.858, 1.480, 0.633, 0.000, 1.084, 2.084, 3.084],
            [7.498, 5.720, 3.942, 2.564, 1.717, 1.084, 0.000, 1.000, 2.000],
            [8.498, 6.720, 4.942, 3.564, 2.717, 2.084, 1.000, 0.000, 1.000],
            [9.498, 7.720, 5.942, 4.564, 3.717, 3.084, 2.000, 1.000, 0.000]
        ]), dtype=torch.float32).to(x.device)  # 确保 device 一致


        # x: (B, N, D), y: (B, M, D)
        return torch.einsum('bnd,dc,bmc->bnm', x, cost_matrix, y)

    def emd_loss(self, source, batch):
        """
        Efficient batched EMD loss computation.

        Args:
            source: Tensor of shape [B, vocab_size] - model logits
            batch: dict with:
                - 'indices': List[List[int]]  (length = B)
                - 'labels': List[List[float]] (length = B)
        Returns:
            scalar EMD loss
        """
        p = 2
        entreg = 0.1
        OTLoss = SamplesLoss(
            loss='sinkhorn',
            p=p,
            blur=entreg ** (1 / p),
            backend='tensorized'
        )
        if source.dtype != torch.float32:
            source = source.float()

        device = source.device
        B = source.size(0)

        # Step 1: map vocab index to token id for the entire batch
        try:
            mapped_token_ids = [
                [self.token_ids_dict[vocab_id] for vocab_id in indices]
                for indices in batch["indices"]
            ]  # List[List[int]], shape [B, K]
        except KeyError as e:
            raise ValueError(f"Token ID mapping failed. Missing key: {e}")

        K = len(mapped_token_ids[0])  # same for all due to bucketed batching
        token_ids_tensor = torch.tensor(mapped_token_ids, device=device)  # [B, K]

        # Step 2: get the logits for these token ids from full vocab logits
        logits_selected = torch.gather(source, dim=1, index=token_ids_tensor)  # [B, K]
        pred_probs = torch.softmax(logits_selected, dim=-1)  # [B, K]

        # Step 3: build soft labels tensor
        label_tensor = torch.tensor(batch["labels"], dtype=torch.float32, device=device)  # [B, K]

        if self.cfg['training']['use_cost']:
            if label_tensor.size(1)==9:
                return self.OTLoss_cost(pred_probs, label_tensor)
            else:
                return self.OTLoss(pred_probs, label_tensor)
            # Step 4: compute batched EMD loss
        else:
        #original
            return OTLoss(pred_probs, label_tensor)


    def triplet_emd_loss(self, anchor_logits, positive_logits, negative_logits,
                         batch,
                         margin=1.0, p=2, entreg=0.1, dynamic_margin=True, k=0.4, tau=1.0):
        """
        Triplet Sinkhorn loss with softplus-based smooth dynamic margin.
        """
        device = anchor_logits.device
        B = anchor_logits.size(0)

        sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=entreg ** (1 / p),
            backend="tensorized"
        )

        def gather_probs(logits, indices_list, label):
            try:
                token_ids = [
                    [self.token_ids_dict[idx] for idx in row]
                    for row in indices_list
                ]
            except KeyError as e:
                raise ValueError(f"[{label}] Missing vocab index mapping: {e}")
            token_ids_tensor = torch.tensor(token_ids, device=device)  # [B, K]
            selected_logits = torch.gather(logits, dim=1, index=token_ids_tensor)  # [B, K]
            probs = torch.softmax(selected_logits.float(), dim=-1)  # [B, K]
            return probs

        a_probs = gather_probs(anchor_logits, batch["anchor_indices"], label="anchor")
        p_probs = gather_probs(positive_logits, batch["positive_indices"], label="positive")
        n_probs = gather_probs(negative_logits, batch["negative_indices"], label="negative")

        emd_ap = sinkhorn(a_probs, p_probs)  # [B]
        emd_an = sinkhorn(a_probs, n_probs)  # [B]

        if dynamic_margin:
            Z = max(emd_ap.max().item(), emd_an.max().item(), 1e-6)
            f_s_ap = 1 - emd_ap / Z
            f_s_an = 1 - emd_an / Z

            f_t_ap = torch.exp(-emd_ap / tau)
            f_t_an = torch.exp(-emd_an / tau)

            Ra = (1 - f_t_an) * f_s_an
            Rb = f_t_ap * (1 - f_s_ap)

            # ✅ Softplus-based smooth margin (no hard clipping)
            diff = Ra - Rb  # [B]
            scale = torch.log1p(torch.exp(torch.tensor(tau, device=device)))  # log(1 + e^tau)
            dyn_margin = k / scale * torch.log1p(torch.exp(diff))  # [B]
        else:
            dyn_margin = margin

        triplet = torch.clamp(emd_ap - emd_an + dyn_margin, min=0.0)  # [B]
        return triplet.mean()



    def compute_loss(self, batch, mode):
        if mode == "duration":
            probs = self.forward(batch["input_ids"], batch["attention_mask"])
            return self.emd_loss(probs, batch)

        elif mode == "triplet":
            a_probs = self.forward(batch["anchor_input_ids"], batch["anchor_attention_mask"])
            p_probs = self.forward(batch["positive_input_ids"], batch["positive_attention_mask"])
            n_probs = self.forward(batch["negative_input_ids"], batch["negative_attention_mask"])
            return self.triplet_emd_loss(a_probs, p_probs, n_probs, batch)

        elif mode == "order":
            probs = self.forward(batch["ord_input_ids"], batch["ord_attention_mask"])
            return self.softce_loss(probs, batch)


        elif mode == "joint":
            probs = self.forward(batch["input_ids"], batch["attention_mask"])
            emd = self.emd_loss(probs, batch["labels"])

            a_probs = self.forward(batch["anchor_input_ids"], batch["anchor_attention_mask"])
            p_probs = self.forward(batch["positive_input_ids"], batch["positive_attention_mask"])
            n_probs = self.forward(batch["negative_input_ids"], batch["negative_attention_mask"])
            triplet = self.triplet_emd_loss(a_probs, p_probs, n_probs)

            return emd + triplet  

        else:
            raise ValueError(f"Unsupported loss mode: {mode}")
