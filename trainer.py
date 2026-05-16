from metric_classification import classification_metrics, find_global_threshold
import numpy as np
import time
from evaluate import eval_single_direction, compute_metrics
from predict import BertPredictor
import glob
import json
import torch
import shutil

import torch.nn as nn
import torch.utils.data

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from doc import Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj, call_model_forward
from metric import accuracy
from models import build_model, ModelOutput, DirectAULoss
from dict_hub import build_tokenizer, get_entity_dict
from logger_config import logger
import os 


class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        # define loss function (criterion) and optimizer
        loss_type = getattr(self.args, 'loss_type', 'infonce')
        self.use_negative_sampling = bool(getattr(self.args, 'use_negative_sampling', True))
        self.use_uniformity_loss = bool(getattr(self.args, 'use_uniformity_loss', False))
        
        self.use_infonce_loss = (loss_type in ['infonce', 'all'])
        self.use_alignment_loss = (loss_type in ['alignment', 'all'])
        if loss_type == 'all':
            self.use_uniformity_loss = True
        
        # Disable negative sampling flags when use_negative_sampling is False
        if not self.use_negative_sampling:
            self.args.pre_batch = 0
            self.args.use_self_negative = False
        
        self.infonce_loss = nn.CrossEntropyLoss().cuda()
        
        if self.use_alignment_loss or self.use_uniformity_loss:
            self.auxiliary_loss = DirectAULoss(
                alpha=getattr(self.args, 'directau_alpha', 1.0),
                gamma=getattr(self.args, 'directau_gamma', 1.0),
                eps=getattr(self.args, 'directau_eps', 1e-12),
                use_alignment=self.use_alignment_loss,
                use_uniformity=self.use_uniformity_loss,
            ).cuda()
        else:
            self.auxiliary_loss = None

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        # tracking fields for loss components
        self.last_regularizer = {'align_loss': 0.0, 'align_loss_scaled': 0.0, 'uniform_loss': 0.0, 'uniform_loss_scaled': 0.0, 'total_aux_loss': 0.0}
        self.last_infonce_loss = 0.0

        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

    def _compute_batch_loss(self, logits, labels, hr_vector, tail_vector, batch_exs, batch_size):
        total_loss = None
        self.last_infonce_loss = 0.0

        if self.use_infonce_loss and self.use_negative_sampling:
            total_loss = self.infonce_loss(logits, labels)
            total_loss = total_loss + self.infonce_loss(logits[:, :batch_size].t(), labels)
            try:
                self.last_infonce_loss = float(total_loss.detach().cpu().item())
            except Exception:
                self.last_infonce_loss = 0.0
        elif self.use_infonce_loss and not self.use_negative_sampling:
            total_loss = self.infonce_loss(logits.diag().unsqueeze(-1), torch.zeros(batch_size, dtype=torch.long, device=logits.device))
            try:
                self.last_infonce_loss = float(total_loss.detach().cpu().item())
            except Exception:
                self.last_infonce_loss = 0.0

        if self.use_alignment_loss or self.use_uniformity_loss:
            regularizer = self.auxiliary_loss(hr_vector, tail_vector, labels, batch_exs=batch_exs)
            total_loss = regularizer['loss'] if total_loss is None else total_loss + regularizer['loss']

            # Store last regularizer components for logging/inspection
            try:
                self.last_regularizer = {
                    'align_loss': float(regularizer.get('align_loss', 0.0).item() if hasattr(regularizer.get('align_loss', 0.0), 'item') else regularizer.get('align_loss', 0.0)),
                    'align_loss_scaled': float(regularizer.get('align_loss_scaled', regularizer.get('align_loss', 0.0)).item() if hasattr(regularizer.get('align_loss_scaled', regularizer.get('align_loss', 0.0)), 'item') else regularizer.get('align_loss_scaled', regularizer.get('align_loss', 0.0))),
                    'uniform_loss': float(regularizer.get('uniform_loss', 0.0).item() if hasattr(regularizer.get('uniform_loss', 0.0), 'item') else regularizer.get('uniform_loss', 0.0)),
                    'uniform_loss_scaled': float(regularizer.get('uniform_loss_scaled', 0.0).item() if hasattr(regularizer.get('uniform_loss_scaled', 0.0), 'item') else regularizer.get('uniform_loss_scaled', 0.0)),
                    'total_aux_loss': float(regularizer.get('loss', 0.0).item() if hasattr(regularizer.get('loss', 0.0), 'item') else regularizer.get('loss', 0.0)),
                }
            except Exception:
                self.last_regularizer = {'align_loss': 0.0, 'align_loss_scaled': 0.0, 'uniform_loss': 0.0, 'uniform_loss_scaled': 0.0, 'total_aux_loss': 0.0}

        if total_loss is None:
            raise RuntimeError('No training objective is enabled; check --loss-type and flags')

        return total_loss

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        total_start_time = time.time()
        train_time = 0.0

        for epoch in range(self.args.epochs):
            epoch_train_start = time.time()
            # train for one epoch
            self.train_epoch(epoch)
            train_time += time.time() - epoch_train_start

            val_start = time.time()
            self._run_eval(epoch=epoch)
            val_time = time.time() - val_start

            # Evaluate MR, MRR, Hits@1/3/10 on valid set using current training model (no second model loaded)
            if self.args.valid_path and self.args.model_dir:
                from dict_hub import get_entity_dict
                from doc import Example, Dataset
                entity_dict = get_entity_dict()
                
                # Use training model directly for entity embeddings
                self.model.eval()
                with torch.no_grad():
                    examples = []
                    for entity_ex in entity_dict.entity_exs:
                        examples.append(Example(head_id='', relation='',
                                                tail_id=entity_ex.entity_id))
                    entity_loader = torch.utils.data.DataLoader(
                        Dataset(path='', examples=examples, task=self.args.task),
                        num_workers=0,
                        batch_size=max(self.args.batch_size, 512),
                        collate_fn=collate,
                        shuffle=False)
                    
                    ent_tensor_list = []
                    for batch_dict in entity_loader:
                        batch_dict['only_ent_embedding'] = True
                        if torch.cuda.is_available():
                            batch_dict = move_to_cuda(batch_dict)
                        outputs = call_model_forward(get_model_obj(self.model), batch_dict)
                        ent_tensor_list.append(outputs['ent_vectors'])
                    entity_tensor = torch.cat(ent_tensor_list, dim=0)
                self.model.train()
                
                # Create a lightweight wrapper for eval_single_direction compatibility
                class ModelWrapper:
                    def __init__(self, model, task, batch_size):
                        self.model = model
                        self.task = task
                        self.batch_size = batch_size
                    
                    def predict_by_examples(self, examples):
                        self.model.eval()
                        with torch.no_grad():
                            data_loader = torch.utils.data.DataLoader(
                                Dataset(path='', examples=examples, task=self.task),
                                num_workers=0,
                                batch_size=max(self.batch_size, 512),
                                collate_fn=collate,
                                shuffle=False)
                            hr_tensor_list, tail_tensor_list = [], []
                            for batch_dict in data_loader:
                                if torch.cuda.is_available():
                                    batch_dict = move_to_cuda(batch_dict)
                                outputs = call_model_forward(get_model_obj(self.model), batch_dict)
                                hr_tensor_list.append(outputs['hr_vector'])
                                tail_tensor_list.append(outputs['tail_vector'])
                        self.model.train()
                        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)
                
                predictor = ModelWrapper(get_model_obj(self.model), self.args.task, self.args.batch_size)
                forward_metrics = eval_single_direction(predictor, entity_tensor, eval_forward=True)
                backward_metrics = eval_single_direction(predictor, entity_tensor, eval_forward=False)
                metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
                log_str = f"[EPOCH {epoch}]\nForward: {json.dumps(forward_metrics)}\nBackward: {json.dumps(backward_metrics)}\nAverage: {json.dumps(metrics)}"
                print(log_str)
                logger.info(log_str)
                with open(os.path.join(self.args.model_dir, 'valid_metrics.log'), 'a', encoding='utf-8') as f:
                    f.write(log_str + '\n')

            # Evaluate triple classification metrics on a separate labeled validation file.
            valid_label_path = self.args.valid_label_path or None
            if valid_label_path is None and self.args.valid_path:
                if self.args.valid_path.endswith('_w_label.txt'):
                    valid_label_path = self.args.valid_path
                elif self.args.valid_path.endswith('.txt'):
                    valid_label_path = self.args.valid_path.replace('.txt', '_w_label.txt')
            if valid_label_path and os.path.exists(valid_label_path):
                # Đọc dữ liệu và label
                from doc import load_data
                valid_exs = load_data(valid_label_path, add_forward_triplet=False, add_backward_triplet=False)
                y_true = [ex.label for ex in valid_exs]
                # Dự đoán xác suất (logit) và nhãn - use existing training model, no second model
                y_prob = []
                batch_size = 128
                self.model.eval()
                with torch.no_grad():
                    for i in range(0, len(valid_exs), batch_size):
                        batch = valid_exs[i:i+batch_size]
                        batch_vec = [ex.vectorize() for ex in batch]
                        batch_dict = collate(batch_vec)
                        if torch.cuda.is_available():
                            for k in batch_dict:
                                if isinstance(batch_dict[k], torch.Tensor):
                                    batch_dict[k] = batch_dict[k].cuda()
                        output_dict = call_model_forward(get_model_obj(self.model), batch_dict)
                        logits = get_model_obj(self.model).compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
                        prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
                        y_prob.extend(prob.tolist())
                self.model.train()
                # Tìm threshold tối ưu trên validation
                threshold = find_global_threshold(y_true, y_prob)
                y_pred = (np.array(y_prob) > threshold).astype(int).tolist()
                metrics_cls = classification_metrics(y_true, y_pred, y_prob)
                log_cls = f"[EPOCH {epoch}] Triple Classification: {json.dumps(metrics_cls)}"
                log_thresh = f"[EPOCH {epoch}] Best threshold on validation: {threshold:.6f}"
                print(log_thresh)
                logger.info(log_thresh)
                log_cls = f"[EPOCH {epoch}] Triple Classification: {json.dumps(metrics_cls)}"
                print(log_cls)
                logger.info(log_cls)
                with open(os.path.join(self.args.model_dir, 'valid_metrics.log'), 'a', encoding='utf-8') as f:
                    f.write(log_thresh + '\n')
                    f.write(log_cls + '\n')

            if (epoch + 1) % 10 == 0 or epoch == self.args.epochs - 1:
                self._run_test_evaluation(epoch)

        self._run_test_evaluation(epoch)
        # # Link prediction evaluation on validation set after each epoch
        # valid_path = self.args.valid_path
        # if valid_path and os.path.exists(valid_path):
        #     from dict_hub import get_entity_dict
        #     entity_dict = get_entity_dict()
        #     log_path = os.path.join(self.args.model_dir, 'valid_linkpred_metrics.log')
        #     self.evaluate_link_prediction_inplace(self.model, valid_path, entity_dict, log_path, eval_forward=True)

        total_time = time.time() - total_start_time
        print(f"[Timing] Training time (s): {round(train_time, 2)}")
        print(f"[Timing] Total run time (s): {round(total_time, 2)}")
        logger.info(f"[Timing] Training time (s): {round(train_time, 2)}")
        logger.info(f"[Timing] Total run time (s): {round(total_time, 2)}")

    def _run_test_evaluation(self, epoch):
        test_results = {}

        test_label_path = os.path.join('data', 'WN18RR', 'test_w_label.txt')
        if self.args.valid_label_path:
            if self.args.valid_label_path.endswith('_w_label.txt'):
                test_label_path = self.args.valid_label_path.replace('valid_w_label.txt', 'test_w_label.txt')
            elif self.args.valid_label_path.endswith('.txt'):
                test_label_path = self.args.valid_label_path.replace('valid.txt', 'test_w_label.txt')
        if test_label_path and os.path.exists(test_label_path):
            log_path = os.path.join(self.args.model_dir, 'test_metrics.log')
            test_results['triple_classification'] = self.evaluate_triple_classification_inplace(
                self.model,
                test_label_path,
                log_path,
            )

        if self.args.valid_path:
            if self.args.valid_path.endswith('.txt.json'):
                test_eval_path = self.args.valid_path.replace('valid.txt.json', 'test.txt.json')
            elif self.args.valid_path.endswith('.txt'):
                test_eval_path = self.args.valid_path.replace('valid.txt', 'test.txt')
            else:
                data_dir = os.path.dirname(self.args.valid_path)
                test_eval_path = os.path.join(data_dir, 'test.txt.json')
        else:
            test_eval_path = os.path.join('data', 'WN18RR', 'test.txt.json')
        if test_eval_path and os.path.exists(test_eval_path):
            test_entity_dict = get_entity_dict()
            test_output_path = os.path.join(self.args.model_dir, 'test_link_prediction.log')
            # Evaluate both forward and backward directions for test set
            forward_metrics = self.evaluate_link_prediction_inplace(self.model, test_eval_path, test_entity_dict, test_output_path, eval_forward=True)
            backward_metrics = self.evaluate_link_prediction_inplace(self.model, test_eval_path, test_entity_dict, test_output_path, eval_forward=False)
            # Average metrics
            if forward_metrics and backward_metrics:
                avg_metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
                log_str = f"[TEST] Forward: {json.dumps(forward_metrics)}\nBackward: {json.dumps(backward_metrics)}\nAverage: {json.dumps(avg_metrics)}"
                print(log_str)
                logger.info(log_str)
                with open(test_output_path, 'a', encoding='utf-8') as f:
                    f.write(log_str + '\n')
                test_results['link_prediction'] = avg_metrics
        if test_results:
            summary = {
                'epoch': epoch,
                'stage': 'test',
                'metrics': test_results,
            }
            summary_path = os.path.join(self.args.model_dir, f'test_metrics_epoch{epoch + 1}.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
            log_str = f"[EPOCH {epoch}] Test summary: {json.dumps(summary, ensure_ascii=False)}"
            print(log_str)
            logger.info(log_str)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metric_dict = self.eval_epoch(epoch)

        # Compute validation link-prediction MRR (average forward+backward) and use it as the "best" criterion
        valid_mrr = None
        valid_eval_path = None
        if getattr(self.args, 'valid_path', None):
            # prefer provided valid_path as-is if it exists
            if os.path.exists(self.args.valid_path):
                valid_eval_path = self.args.valid_path
            else:
                # try common variants
                if self.args.valid_path.endswith('.txt.json'):
                    cand = self.args.valid_path
                elif self.args.valid_path.endswith('.txt'):
                    cand = self.args.valid_path
                else:
                    cand = self.args.valid_path
                if os.path.exists(cand):
                    valid_eval_path = cand

        if valid_eval_path and os.path.exists(valid_eval_path):
            valid_entity_dict = get_entity_dict()
            valid_output_path = os.path.join(self.args.model_dir, 'valid_link_prediction.log')
            forward_metrics = self.evaluate_link_prediction_inplace(self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=True)
            backward_metrics = self.evaluate_link_prediction_inplace(self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=False)
            if forward_metrics and backward_metrics:
                try:
                    valid_mrr = round((forward_metrics.get('mrr', 0) + backward_metrics.get('mrr', 0)) / 2, 4)
                except Exception:
                    valid_mrr = None

        is_best = (valid_mrr is not None) and (self.best_metric is None or valid_mrr > self.best_metric.get('mrr', -1))
        if is_best:
            self.best_metric = {'mrr': valid_mrr}

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = call_model_forward(self.model, batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            hr_vector, tail_vector = outputs.hr_vector, outputs.tail_vector
            
            batch_exs = batch_dict.get('batch_data', None)
            loss = self._compute_batch_loss(logits, labels, hr_vector, tail_vector, batch_exs, batch_size)
            
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        align_meter = AverageMeter('Align', ':.6f')
        uniform_meter = AverageMeter('Uniform', ':.6f')
        infonce_meter = AverageMeter('InfoNCE', ':.6f')
        gradnorm_meter = AverageMeter('GradNorm', ':.4f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3, align_meter, uniform_meter, infonce_meter, gradnorm_meter],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = call_model_forward(self.model, batch_dict)
            else:
                outputs = call_model_forward(self.model, batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            hr_vector, tail_vector = outputs.hr_vector, outputs.tail_vector
            assert logits.size(0) == batch_size
            
            batch_exs = batch_dict.get('batch_data', None)
            loss = self._compute_batch_loss(logits, labels, hr_vector, tail_vector, batch_exs, batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t.item(), 1)
            losses.update(loss.item(), batch_size)

            # Update auxiliary component meters if available
            if hasattr(self, 'last_regularizer') and self.last_regularizer is not None:
                align_meter.update(self.last_regularizer.get('align_loss_scaled', self.last_regularizer.get('align_loss', 0.0)), batch_size)
                # display the gamma-scaled uniformity (actual contribution to loss)
                uniform_meter.update(self.last_regularizer.get('uniform_loss_scaled', self.last_regularizer.get('uniform_loss', 0.0)), batch_size)
            # Update InfoNCE meter
            try:
                infonce_meter.update(getattr(self, 'last_infonce_loss', 0.0), batch_size)
            except Exception:
                pass

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                gradnorm_meter.update(float(grad_norm), 1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                gradnorm_meter.update(float(grad_norm), 1)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)

    def evaluate_triple_classification_inplace(self, model, label_file, output_log_path, batch_size=128):
        import numpy as np
        import json
        import torch
        import os
        from doc import load_data
        from metric_classification import classification_metrics, find_global_threshold
        model.eval()
        if not os.path.exists(label_file):
            print(f"[EVAL] {label_file} not found, skip evaluation.")
            return
        eval_set = 'TEST' if 'test' in label_file else 'VALID'
        print(f"\n[{eval_set}] Evaluating triple classification inplace on {label_file} ...")
        eval_exs = load_data(label_file, add_forward_triplet=False, add_backward_triplet=False)
        y_true = [ex.label for ex in eval_exs]
        y_prob = []
        with torch.no_grad():
            for i in range(0, len(eval_exs), batch_size):
                batch = eval_exs[i:i+batch_size]
                batch_vec = [ex.vectorize() for ex in batch]
                batch_dict = collate(batch_vec)
                if torch.cuda.is_available():
                    for k in batch_dict:
                        if isinstance(batch_dict[k], torch.Tensor):
                            batch_dict[k] = batch_dict[k].cuda()
                    model.cuda()
                output_dict = call_model_forward(model, batch_dict)
                logits = get_model_obj(model).compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
                prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
                y_prob.extend(prob.tolist())
        threshold = find_global_threshold(y_true, y_prob)
        y_pred = (np.array(y_prob) > threshold).astype(int).tolist()
        metrics_cls = classification_metrics(y_true, y_pred, y_prob)
        log_thresh = f"[{eval_set}] Best threshold: {threshold:.6f}"
        log_cls = f"[{eval_set}] Triple Classification: {json.dumps(metrics_cls)}"
        print(log_thresh)
        print(log_cls)
        logger.info(log_thresh)
        logger.info(log_cls)
        with open(output_log_path, 'a', encoding='utf-8') as f:
            f.write(log_thresh + '\n')
            f.write(log_cls + '\n')
        return {
            'threshold': float(threshold),
            'metrics': metrics_cls,
        }

    def evaluate_link_prediction_inplace(self, model, eval_path, entity_dict, output_log_path, batch_size=128, eval_forward=True):
        import torch
        import json
        from doc import load_data, Dataset, collate, Example
        from utils import move_to_cuda
        model.eval()
        if not os.path.exists(eval_path):
            print(f"[EVAL] {eval_path} not found, skip link prediction evaluation.")
            return
        eval_set = 'TEST' if 'test' in eval_path else 'VALID'
        print(f"\n[{eval_set}] Evaluating link prediction inplace on {eval_path} ...")
        examples = load_data(eval_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)
        hr_vectors, _ = [], []
        with torch.no_grad():
            data_loader = torch.utils.data.DataLoader(
                Dataset(path='', examples=examples, task=self.args.task),
                num_workers=0,
                batch_size=batch_size,
                collate_fn=collate,
                shuffle=False)

            for batch_dict in data_loader:
                if torch.cuda.is_available():
                    batch_dict = move_to_cuda(batch_dict)
                    model.cuda()
                outputs = call_model_forward(model, batch_dict)
                hr_vectors.append(outputs['hr_vector'])
        hr_tensor = torch.cat(hr_vectors, dim=0)
        entity_examples = [Example(head_id='', relation='', tail_id=entity_ex.entity_id) for entity_ex in entity_dict.entity_exs]
        entity_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=entity_examples, task=self.args.task),
            num_workers=0,
            batch_size=max(batch_size, 512),
            collate_fn=collate,
            shuffle=False)

        entity_vectors = []
        for batch_dict in entity_loader:
            batch_dict['only_ent_embedding'] = True
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
                model.cuda()
            outputs = call_model_forward(model, batch_dict)
            entity_vectors.append(outputs['ent_vectors'])

        entities_tensor = torch.cat(entity_vectors, dim=0)
        if torch.cuda.is_available():
            hr_tensor = hr_tensor.cuda()
            entities_tensor = entities_tensor.cuda()
        target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
        chunk_size = getattr(self.args, 'chunk_size', 8192)
        topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entities_tensor, target=target, examples=examples, batch_size=batch_size, chunk_size=chunk_size)
        log_str = f"[{eval_set}] Link Prediction Metrics: {json.dumps(metrics)}"
        print(log_str)
        logger.info(log_str)
        with open(output_log_path, 'a', encoding='utf-8') as f:
            f.write(log_str + '\n')
        return metrics
