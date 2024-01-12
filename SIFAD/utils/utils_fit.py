import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


#---------------------------------------#
#   冻结其他模块
#---------------------------------------#
def freeze_parameters_except_ACCR(model):
    for name, param in model.named_parameters():
        if 'accr' not in name:
            param.requires_grad = False
    for param in model.accr.parameters():
        param.requires_grad = True

#---------------------------------------#
#   冻结ACCR模块
#---------------------------------------#
def freeze_ACCR_regularizer(model):
    for param in model.accr.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'accr' not in name:
            param.requires_grad = True

    for param in model.backbone.parameters():
        param.requires_grad = False


#---------------------------------------#
#   梯度翻转
#---------------------------------------#
def flip_grads(mod):
    for para in mod.parameters():
        if para.requires_grad:
            para.grad = - para.grad


def fit_one_epoch(model_train, model, ema, accr_loss, yolo_loss, loss_history, eval_callback, optimizer, optACCR, epoch,
                  Freeze_Epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir,
                  local_rank=0):
    loss = 0
    val_loss = 0
    train_ACCR = False
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        # ---------------------------------------#
        #   每30轮训练ACCR, 50轮训练其他
        # ---------------------------------------#
        # print(model)
        if epoch > Freeze_Epoch:
            if iteration % 80 < 30:
                freeze_parameters_except_ACCR(model)
                train_ACCR = True
            else:
                freeze_ACCR_regularizer(model)
                train_ACCR = False
        # print(batch)
        images, targets, labels_scale = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                labels_scale = labels_scale.cuda(local_rank)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        if epoch > Freeze_Epoch:
            if train_ACCR:
                optACCR.zero_grad()
            else:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, rcho, scale_logits = model_train(images)
            loss_value = yolo_loss(outputs, targets, images)
            loss_accr = accr_loss(scale_logits, labels_scale, rcho)
            loss_value = loss_value + loss_accr

            # ----------------------#
            #   反向传播
            # ----------------------#
            if epoch >= Freeze_Epoch:
                if train_ACCR:
                    # print('Training ACCR...')
                    loss_value.backward()
                    flip_grads(model.accr)
                    optACCR.step()
                else:
                    # print('Training YOLOv7...')
                    loss_value.backward()
                    optimizer.step()
            else:
                loss_value.backward()
                optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs, rcho, scale_logits = model_train(images)
                loss_value = yolo_loss(outputs, targets, images)
                loss_accr = accr_loss(scale_logits, labels_scale, rcho)
                loss_value = loss_value + loss_accr

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, labels_scale = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                labels_scale = labels_scale.cuda(local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, rcho, scale_logits = model_train(images)
            loss_value = yolo_loss(outputs, targets, images)
            loss_accr = accr_loss(scale_logits, labels_scale, rcho)
            loss_value = loss_value + loss_accr

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val, loss_accr / epoch_step)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f || ACCR Loss: %.3f' % (loss / epoch_step, val_loss / epoch_step_val
                                                                         , loss_accr / epoch_step))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

def fit_one_epoch_noACCR(model_train, model, ema, accr_loss, yolo_loss, loss_history, eval_callback, optimizer, epoch,
                  epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir,
                  local_rank=0):
    loss = 0
    val_loss = 0
    # train_ACCR = True
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        # print(model)
        # print(batch)
        images, targets, labels_scale = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                labels_scale = labels_scale.cuda(local_rank)
        # ----------------------#
        #   ????????
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   ????????
            # ----------------------#
            outputs, rcho, scale_logits = model_train(images)
            loss_value = yolo_loss(outputs, targets, images)
            loss_accr = accr_loss(scale_logits, labels_scale, rcho)
            loss_value = loss_value + loss_accr

            # ----------------------#
            #   ????????
            # ----------------------#

            # loss_value.requires_grad_(True)
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   ????????
                # ----------------------#
                outputs, rcho, scale_logits = model_train(images)
                loss_value = yolo_loss(outputs, targets, images)
                loss_accr = accr_loss(scale_logits, labels_scale, rcho)
                loss_value = loss_value + loss_accr

            # ----------------------#
            #   ????????
            # ----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, labels_scale = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                labels_scale = labels_scale.cuda(local_rank)
            # ----------------------#
            #   ????????
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   ????????
            # ----------------------#
            outputs, rcho, scale_logits = model_train(images)
            loss_value = yolo_loss(outputs, targets, images)
            loss_accr = accr_loss(scale_logits, labels_scale, rcho)
            loss_value = loss_value + loss_accr

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val, loss_accr / epoch_step)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f || ACCR Loss: %.3f' % (loss / epoch_step, val_loss / epoch_step_val
                                                                         , loss_accr / epoch_step))

        # -----------------------------------------------#
        #   ????????
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))