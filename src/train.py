from torch.optim import AdamW
from torch.optim import RMSprop
from torch.optim import Adam
from transformers import get_scheduler
from tqdm import tqdm
import os
from itertools import cycle
from .evaluate import evaluate_on_mctaco, evaluate_model, evaluate_on_situatedgen, evaluate_on_td

import sys

def train_model(model,
                tokenizer,
                output_dir,
                config,
                train_dataloader=None,
                eval_dataloader=None,
                train_contra_dataloader=None,
                train_ord_dataloader=None,
                train_qa_dataloader=None,
                train_gen_dataloader=None,
                ):
    lr=float(config['training']['learning_rate'])
    num_epochs = config['training']['num_epochs']
    weight_decay = float(config['training']['weight_decay'])
    max_grad_norm = float(config['training']['max_grad_norm'])
    log_every = config['training']['log_every']
    run_eval_before_train = config['training']['run_eval_before_train']
    num_warmup_steps=config['training']['warm_up']
    train_mode=config['training']['mode']
    emd_weight=config['training']['emd_weight']
    triplet_weight=config['training']['triplet_weight']
    order_weight=config['training']['order_weight']
    qa_weight=config['training']['qa_weight']
    margin = float(config['training']['margin'])  
    evaluate_few_shot=config['evaluation']['evaluate_few_shot']
    train_gen = config["training"]["train_gen"]

    model.margin = margin  


    device = model.model.device
    best_performance = 0.0
    best_performance_td = 0.0
    best_performance_dq = 0.0
    best_performance_cot = 0.0

    if train_dataloader is not None:
        print(f"[INFO] Duration dataset: {len(train_dataloader.dataset)} samples, {len(train_dataloader)} batches")
    else:
        print("[INFO] Duration dataset: None")

    if train_contra_dataloader is not None:
        print(f"[INFO] Triplet dataset: {len(train_contra_dataloader.dataset)} samples, {len(train_contra_dataloader)} batches")
    else:
        print("[INFO] Triplet dataset: None")

    if train_ord_dataloader is not None:
        print(f"[INFO] Ordering dataset: {len(train_ord_dataloader.dataset)} samples, {len(train_ord_dataloader)} batches")
    else:
        print("[INFO] Ordering dataset: None")

    if train_qa_dataloader is not None:
        print(f"[INFO] QA dataset: {len(train_qa_dataloader.dataset)} samples, {len(train_qa_dataloader)} batches")
    else:
        print("[INFO] QA dataset: None")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    if run_eval_before_train:

        if train_gen:
            print("Evaluating untrained model on MC-TACO...")
            print("[Situated Gen] ========================================Initial Performance: ========================================")
            performance = evaluate_on_situatedgen(model, tokenizer, device, epoch=-1, output_dir=output_dir, config=config)
            tqdm.write(f"[Situated Gen] Initial Performance: "
                       f"BLEU-4={performance['BLEU-4']:.4f}, "
                       f"METEOR={performance['METEOR']:.4f}, "
                       f"CIDEr={performance['CIDEr']:.4f}, "
                       f"SPICE={performance['SPICE']:.4f}, "
                       f"MATCH={performance['MATCH']:.4f}, "
                       f"S={performance['S']:.4f}")

        else:

            print("Evaluating untrained model on MC-TACO...")
            print("[MC-TACO] ========================================Initial Performance: ========================================")
            initial_metrics = evaluate_on_mctaco(model, tokenizer, device, epoch=-1, output_dir=output_dir, config=config)
            print(f"[MC-TACO] Initial Performance: "
                  f"acc={initial_metrics['acc']:.4f}, "
                  f"strict={initial_metrics['strict_acc']:.4f}, "
                  f"avg_f1={initial_metrics['avg_f1']:.4f}, "
                  f"micro_f1={initial_metrics['micro_f1']:.4f}")


            if evaluate_few_shot:
                print("[MC-TACO] ========================================Initial Few-shot Performance: ========================================")
                original_num_shots = config["evaluation"]["num_shots"]
                config["evaluation"]["num_shots"]=3
                initial_metrics = evaluate_on_mctaco(model, tokenizer, device, epoch=-1, output_dir=output_dir, config=config)
                config["evaluation"]["num_shots"]=original_num_shots
                print(f"[MC-TACO] Initial Few-shot Performance: "
                      f"acc={initial_metrics['acc']:.4f}, "
                      f"strict={initial_metrics['strict_acc']:.4f}, "
                      f"avg_f1={initial_metrics['avg_f1']:.4f}, "
                      f"micro_f1={initial_metrics['micro_f1']:.4f}")

  

    optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    if train_mode == 'duration_only':
        total_steps = num_epochs * len(train_dataloader)
    elif train_mode == 'triplet_only':
        total_steps = num_epochs * len(train_contra_dataloader)
    elif train_mode == 'combined_ord':
        total_steps = num_epochs * max(len(train_dataloader), len(train_contra_dataloader), len(train_ord_dataloader))
    
    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=num_warmup_steps,
                                 num_training_steps=total_steps)

    global_step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0.0
        if train_mode == 'combined':
            steps_per_epoch = max(len(train_dataloader), len(train_contra_dataloader))
            emd_iter = cycle(train_dataloader) if len(train_dataloader) < len(train_contra_dataloader) else iter(train_dataloader)
            triplet_iter = cycle(train_contra_dataloader) if len(train_contra_dataloader) < len(train_dataloader) else iter(train_contra_dataloader)

            for step in tqdm(range(steps_per_epoch), desc="Combined loss training"):
                emd_batch = next(emd_iter)
                triplet_batch = next(triplet_iter)

                merged_batch = {**emd_batch, **triplet_batch}

                loss_emd = model.compute_loss(merged_batch, mode="duration")
                loss_triplet = model.compute_loss(merged_batch, mode="triplet")
                loss = emd_weight * loss_emd + triplet_weight * loss_triplet

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                global_step += 1
                if global_step % log_every == 0:
                    tqdm.write(f"Step {global_step}, Loss: {loss.item():.4f}")

        elif train_mode == 'combined_ord':
            steps_per_epoch = max(len(train_dataloader), len(train_contra_dataloader), len(train_ord_dataloader))

            emd_iter = cycle(train_dataloader) if len(train_dataloader) < steps_per_epoch else iter(train_dataloader)
            triplet_iter = cycle(train_contra_dataloader) if len(train_contra_dataloader) < steps_per_epoch else iter(
                train_contra_dataloader)
            order_iter = cycle(train_ord_dataloader) if len(train_ord_dataloader) < steps_per_epoch else iter(
                train_ord_dataloader)

            for step in tqdm(range(steps_per_epoch), desc="Combined ORD loss training"):
                emd_batch = next(emd_iter)
                triplet_batch = next(triplet_iter)
                order_batch = next(order_iter)

                merged_batch = {**emd_batch, **triplet_batch, **order_batch}

                loss_emd = model.compute_loss(merged_batch, mode="duration")
                loss_triplet = model.compute_loss(merged_batch, mode="triplet")
                loss_order = model.compute_loss(merged_batch, mode="order")

                loss = emd_weight * loss_emd + triplet_weight * loss_triplet + order_weight * loss_order

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1
                if global_step % log_every == 0:
                    tqdm.write(
                        f"Step {global_step}, Total Loss: {loss.item():.4f}, EMD: {loss_emd.item():.4f}, Triplet: {loss_triplet.item():.4f}, Order: {loss_order.item():.4f}")



        else:
            raise ValueError("Invalid train_mode. Choose from 'duration_only', 'triplet_only', 'joint', 'combined'")

        avg_epoch_loss = epoch_loss / max(len(train_dataloader), len(train_contra_dataloader))
        tqdm.write(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

		model_dir = os.path.join(output_dir, "best_model_cot")
        performance = evaluate_on_mctaco(model, tokenizer, device, epoch, output_dir, config)
            tqdm.write(f"[MCTACO] Epoch {epoch + 1} Performance: "
                       f"acc={performance['acc']:.4f}, "
                       f"strict={performance['strict_acc']:.4f}, "
                       f"avg_f1={performance['avg_f1']:.4f}, "
                       f"micro_f1={performance['micro_f1']:.4f}")
            if performance['acc'] > best_performance:
                best_performance = performance['acc']
                best_model_dir = os.path.join(output_dir, "best_model")
                model.model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                tqdm.write(f"[MC-TACO] Best model saved with accuracy: {best_performance:.4f}")
        







