from tqdm import tqdm
import os
import numpy as np
from tensorboardX import SummaryWriter
from .utils import AverageMeter, save_checkpoint, evaluate_seg
import torch
from .metrics import DiceScoreStorer, IoUStorer
import warnings

warnings.filterwarnings("ignore")


def train_clcc(_print, cfg, model, train_loader, valid_loader, criterion, valid_criterion, optimizer,
               scheduler, start_epoch, best_metric, test_loader):
    _print('train train_clcc')

    cont_loss = criterion[1]
    cons_loss = criterion[2]

    tb = SummaryWriter(f"runs/{cfg.EXP}/{cfg.MODEL.NAME}", comment=f"{cfg.COMMENT}")

    labeled_idxes = set()
    unlabeled_idxes = set()

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"Epoch {epoch + 1}")

        # define some meters
        losses = AverageMeter()
        losses_sup = AverageMeter()
        losses_cont_g = AverageMeter()
        losses_gc = AverageMeter()

        top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
        top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)

        """
        TRAINING
        """
        model.train()
        tbar = tqdm(train_loader)

        for i, batch in enumerate(tbar):
            image = batch['image']
            target = batch['label']
            p_image = batch['image']
            criterion_res = criterion[0]
            bs = image.size()[0]
            imidxes = batch['imidx']
            for i in range(len(imidxes)):
                if i < cfg.TRAIN.LB_BATCH_SIZE:
                    labeled_idxes.add(imidxes[i].item())
                else:
                    unlabeled_idxes.add(imidxes[i].item())

            image = image.to(device='cuda', dtype=torch.float)
            target = target.to(device='cuda', dtype=torch.float)[:cfg.TRAIN.LB_BATCH_SIZE]
            p_image = p_image.to(device='cuda', dtype=torch.float)

            outputs = model(image, output_final_feat=True)
            output_target = outputs['seg_final']
            proj_final = outputs["proj_final"]

            cn, cs = cfg.MODEL.PROJECT_NUM, cfg.DATA.SIZE // cfg.MODEL.PROJECT_NUM
            p_image = p_image.unfold(2, cs, cs).unfold(3, cs, cs).permute(
                0, 2, 3, 1, 4, 5).reshape(-1, cfg.DATA.INP_CHANNELS, cs, cs)
            p_output = model(p_image, output_final_feat=True)
            p_proj_final = p_output["proj_final"]
            p_output_target = p_output['seg_final']
            p_proj_final = p_proj_final.reshape(bs, cn, cn, 512 // cfg.MODEL.FEATURE_SCALE, 1, 1).permute(
                0, 3, 1, 4, 2, 5).reshape(bs, 512 // cfg.MODEL.FEATURE_SCALE, cn, cn)

            loss_sup = criterion_res(output_target[:cfg.TRAIN.LB_BATCH_SIZE], target[:cfg.TRAIN.LB_BATCH_SIZE])

            if epoch < cfg.OPT.WARM_UP:
                global_consistency_loss = torch.tensor(0.0).cuda()
                global_cont_loss = cont_loss(proj_final, p_proj_final)
            else:
                global_cont_loss = torch.tensor(0.0).cuda()
                global_consistency_loss = cons_loss(p_output_target, output_target.detach())

            if epoch < cfg.OPT.WARM_UP:
                consistency_weight = cfg.OPT.MAX_C
                loss = consistency_weight * global_cont_loss + loss_sup
            else:
                consistency_weight = cfg.OPT.MAX_C
                loss = loss_sup + consistency_weight * global_consistency_loss

            top_dice.update(output_target[:cfg.TRAIN.LB_BATCH_SIZE], target[:cfg.TRAIN.LB_BATCH_SIZE])
            top_iou.update(output_target[:cfg.TRAIN.LB_BATCH_SIZE], target[:cfg.TRAIN.LB_BATCH_SIZE])

            loss = loss / cfg.OPT.GD_STEPS

            loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                scheduler(optimizer, i, epoch, None)  # Cosine LR Scheduler
                optimizer.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            losses_sup.update(loss_sup.item() * cfg.OPT.GD_STEPS, cfg.TRAIN.LB_BATCH_SIZE)
            losses_cont_g.update(global_cont_loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            losses_gc.update(global_consistency_loss * cfg.OPT.GD_STEPS, image.size(0))

            tbar.set_description(
                "Train iou: %.3f, dice: %.3f loss: %.3f (%.5f + (%.5f + %.5f) * %.5f)" % (
                    top_iou.avg, top_dice.avg, losses.avg, losses_sup.avg, losses_cont_g.avg, losses_gc.avg,
                    consistency_weight))

            tb.add_scalars('Loss_res', {'loss': losses.avg,
                                        'losses_sup': losses_sup.avg,
                                        'consistency_weight': consistency_weight}, epoch)
            tb.add_scalars('Train_res',
                           {'top_dice_res': top_dice.avg,
                            'top_iou_res': top_iou.avg}, epoch)
            tb.add_scalars('Lr', {'Lr': optimizer.param_groups[-1]['lr']}, epoch)

        _print("Train iou: %.3f, dice: %.3f, loss: %.3f" % (top_iou.avg, top_dice.avg, losses.avg))

        """
        VALIDATION
        """
        if (epoch + 1) % cfg.VAL.EPOCH == 0:
            top_dice_valid, top_iou_valid = valid_model(_print, cfg, model, valid_criterion, valid_loader)

            # Take dice_score as main_metric to save checkpoint
            is_best = top_dice_valid > best_metric
            best_metric = max(top_dice_valid, best_metric)

            # tensorboard
            if cfg.DEBUG == False:
                tb.add_scalars('Valid',
                               {'top_dice': top_dice_valid,
                                'top_iou': top_iou_valid}, epoch)

                save_checkpoint({
                    "epoch": epoch + 1,
                    "arch": cfg.EXP,
                    "state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "optimizer": optimizer.state_dict(),
                }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}_{cfg.MODEL.NAME}_fold{cfg.TRAIN.FOLD}.pth")

    assert len(labeled_idxes & unlabeled_idxes) == 0

    # test_model(_print, cfg, model, test_loader)
    test_model(_print, cfg, model, test_loader,
               weight=os.path.join(cfg.DIRS.WEIGHTS, f"best_{cfg.EXP}_{cfg.MODEL.NAME}_fold{cfg.TRAIN.FOLD}.pth"))

    if cfg.DEBUG == False:
        # #export stats to json
        tb.export_scalars_to_json(
            os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}_{cfg.MODEL.NAME}_{cfg.COMMENT}_{round(best_metric, 4)}.json"))
        # #close tensorboard
        tb.close()


def valid_model(_print, cfg, model, valid_criterion, valid_loader):
    losses = AverageMeter()
    top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
    top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)

    model.eval()
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, batch in enumerate(tbar):
            image = batch['image']
            target = batch['label']

            image = image.to(device='cuda', dtype=torch.float)
            target = target.to(device='cuda', dtype=torch.float)

            output = model(image)
            output_target = output["seg_final"]

            loss = valid_criterion[0](output_target, target)
            top_dice.update(output_target, target)
            top_iou.update(output_target, target)

            # record
            losses.update(loss.item(), image.size(0))

    _print("Valid iou: %.3f, dice: %.3f loss: %.3f" % (top_iou.avg, top_dice.avg, losses.avg))

    return top_dice.avg, top_iou.avg


def test_model(_print, cfg, model, test_loader, weight=''):
    if weight != '':
        model.load_state_dict(torch.load(weight)["state_dict"])

    model.eval()
    tbar = tqdm(test_loader)
    MAE = []
    Recall = []
    Precision = []
    Accuracy = []
    Dice = []
    IoU_polyp = []

    save_dirs = os.path.join(cfg.DIRS.TEST, cfg.EXP)
    os.makedirs(save_dirs, exist_ok=True)
    os.makedirs(os.path.join(save_dirs, 'pred'), exist_ok=True)
    with torch.no_grad():
        for batch in tbar:
            _id = batch['imidx']
            image = batch['image']
            target = batch['label']

            image = image.to(device='cuda', dtype=torch.float)
            target = target.to(device='cuda', dtype=torch.float)

            output = model(image)["seg_final"]

            out_evl = evaluate_seg(output.permute(0, 2, 3, 1).squeeze(), target.squeeze())

            MAE.append(out_evl[0])
            Recall.append(out_evl[1])
            Precision.append(out_evl[2])
            Accuracy.append(out_evl[3])
            Dice.append(out_evl[4])
            IoU_polyp.append(out_evl[5])

    _print('=========================================')
    _print('MAE: %.3f' % np.mean(MAE))
    _print('Recall: %.3f' % np.mean(Recall))
    _print('Precision: %.3f' % np.mean(Precision))
    _print('Accuracy: %.3f' % np.mean(Accuracy))
    _print('Dice: %.3f' % np.mean(Dice))
    _print('IoU_polyp: %.3f' % np.mean(IoU_polyp))
