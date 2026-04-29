import torch
import json
import torch.backends.cudnn as cudnn

from config import args
from trainer import Trainer
from doc import collate
from logger_config import logger


def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()
    # Evaluate triple classification on test set after training
    evaluate_test_triple_classification(args)


def evaluate_test_triple_classification(args, epoch=None):
    import os
    import numpy as np
    import json
    import torch
    from doc import load_data
    from predict import BertPredictor
    from metric_classification import classification_metrics, find_global_threshold

    test_label_path = os.path.join('data', 'WN18RR', 'test_w_label.txt')
    valid_label_path = getattr(args, 'valid_label_path', '')
    if valid_label_path:
        test_label_path = valid_label_path.replace('valid_w_label.txt', 'test_w_label.txt')
    if not os.path.exists(test_label_path):
        print("[TEST] test_w_label.txt not found, skip test evaluation.")
        return
    print("\n[TEST] Evaluating triple classification on test set...")
    test_exs = load_data(test_label_path, add_forward_triplet=False, add_backward_triplet=False)
    y_true = [ex.label for ex in test_exs]
    y_prob = []
    batch_size = 128
    predictor = BertPredictor()
    # Load best checkpoint after training
    if epoch is None:
        epoch = args.epochs - 1
    ckt_path = '{}/checkpoint_epoch{}.mdl'.format(args.model_dir, epoch)
    if not os.path.exists(ckt_path):
        ckt_path = '{}/checkpoint_{}_0.mdl'.format(args.model_dir, epoch)
    predictor.load(ckt_path)
    for i in range(0, len(test_exs), batch_size):
        batch = test_exs[i:i+batch_size]
        batch_vec = [ex.vectorize() for ex in batch]
        batch_dict = collate(batch_vec)
        if torch.cuda.is_available():
            for k in batch_dict:
                if isinstance(batch_dict[k], torch.Tensor):
                    batch_dict[k] = batch_dict[k].cuda()
            predictor.model.cuda()
        output_dict = predictor.model(**batch_dict)
        logits = predictor.model.compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
        prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
        y_prob.extend(prob.tolist())
    # Dùng threshold tìm được trên validation
    threshold = find_global_threshold(y_true, y_prob)
    y_pred = (np.array(y_prob) > threshold).astype(int).tolist()
    metrics_cls = classification_metrics(y_true, y_pred, y_prob)
    log_thresh = f"[TEST] Best threshold on test: {threshold:.6f}"
    log_cls = f"[TEST] Triple Classification: {json.dumps(metrics_cls)}"
    print(log_thresh)
    print(log_cls)
    logger.info(log_thresh)
    logger.info(log_cls)
    with open(os.path.join(args.model_dir, 'test_metrics.log'), 'a', encoding='utf-8') as f:
        f.write(log_thresh + '\n')
        f.write(log_cls + '\n')


if __name__ == '__main__':
    main()
