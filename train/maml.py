import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train.utils.utils import tensorboard_batch_figure
from train.utils.utils import IGBP_simplified_classes
from train.utils.data import prepare_dataset
from train.utils.utils import prepare_transform_and_model
from train.utils.utils import update_parameters
import torchmeta

MODEL_URL = "https://bagofmaml.s3.eu-central-1.amazonaws.com/app/model.pth"

def reset_indices(targets):
    """
    resets absolute class indices (1,7,5,3) with relative ones (0,1,2,3)
    """
    rows = []
    for row in targets:
        class_ids = row.unique()

        for idx, id in enumerate(class_ids):
            row[row == id] = idx
        rows.append(row)
    return torch.stack(rows)

def train(args):
    classes = IGBP_simplified_classes

    transform, model = prepare_transform_and_model(args)

    if args.start_from_pretrained:
        print(f"loading model from {MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")

        # remove "module." from keys (its added from parallel computing)
        state_dict = OrderedDict({k.replace("module.",""):v for k,v in state_dict.items()})
        model.load_state_dict(state_dict)

    dataloader, valdataloader, testdataloader = prepare_dataset(args, transform)
    valdataloader_enumerate = enumerate(valdataloader)
    testdataloader_enumerate = enumerate(testdataloader)

    if args.not_one_vs_all:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.num_ways))

    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_step_size, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, patience=args.patience // 2)

    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1e-1, 1e-1))

    summary_writer = SummaryWriter(
        log_dir=os.path.join(args.output_folder, "tensorboard"))

    modelpath = os.path.join(args.output_folder, f"model_best.pth")
    if os.path.exists(modelpath):
        state_dict = torch.load(os.path.join(args.output_folder, f"model_best.pth"))
        state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)
        meta_optimizer.load_state_dict(torch.load(os.path.join(args.output_folder, f"optimizer_best.pth")))
        f = pd.read_csv(os.path.join(args.output_folder, f"log.csv"), index_col=0)
        start_batch_idx = int(df.iloc[-1].episode)
        stats = df.to_dict(orient='list')
        stats = [dict(zip(stats,t)) for t in zip(*stats.values())]
        print(f"resuming from {modelpath} episode {start_batch_idx}")
    else:
        start_batch_idx = 0
        stats = []

    if torch.cuda.device_count() > 1:
        model = torchmeta.modules.DataParallel(model)

    # Training loop
    not_improved_counter = 0 # early stopping counter
    trainacc, valacc, trainloss, valloss = [], [], [], []
    with tqdm(dataloader, total=args.num_batches - start_batch_idx) as pbar:
        for batch_idx, batch in enumerate(pbar):
            batch_idx += start_batch_idx

            trainloss_, train_predictions_, train_targets_, train_ids = meta_train_episode(batch, model,
                                                                                          meta_optimizer,
                                                                                          criterion,
                                                                                          args)
            trainloss.append(trainloss_.cpu().detach())
            trainacc.append(sklearn.metrics.accuracy_score(train_targets_.cpu().view(-1), train_predictions_.cpu().view(-1)))

            # first train only then start validating
            if batch_idx < args.validate_after_n_episodes:
                pbar.set_description_str(f"l={np.stack(trainloss).mean():.2f}, acc={np.stack(trainacc).mean():.2f}. ({batch_idx}/{args.validate_after_n_episodes} until validation)")
                continue

            test_idx, test_batch = next(valdataloader_enumerate)
            valloss_, val_predictions_, val_targets_, val_id_ = meta_test_episode(test_batch, model, criterion, args)
            valloss.append(valloss_.cpu().detach())
            valacc.append(sklearn.metrics.accuracy_score(val_targets_.cpu().view(-1), val_predictions_.cpu().view(-1)))
            pbar.set_description_str(
                f"l={np.stack(trainloss).mean():.2f}|{np.stack(valloss).mean():.2f}, acc={np.stack(trainacc).mean():.2f}|{np.stack(valacc).mean():.2f}")


            if (batch_idx % args.log_every_n_batches == 0 or batch_idx >= args.num_batches) and (len(trainacc) >= args.log_every_n_batches):

                global_step = batch_idx * args.batch_size

                summary_writer.add_scalars("loss", dict(trainloss=np.stack(trainloss).mean(), testloss=np.stack(valloss).mean()), batch_idx)
                summary_writer.add_scalars("accuracy", dict(trainaccuracy=np.stack(trainacc).mean(), valaccuracy=np.stack(valacc).mean()), batch_idx)

                stats.append(
                    dict(
                        episode=batch_idx,
                        samples=global_step,
                        trainloss=np.stack(trainloss).mean(),
                        valloss=np.stack(valloss).mean(),
                        trainacc=np.stack(trainacc).mean(),
                        valacc=np.stack(valacc).mean(),
                    )
                )
                scheduler.step(stats[-1]["valloss"])
                trainacc, valacc, trainloss, valloss = [], [], [], []


                if args.tensorboard_log_images:
                    tensorboard_batch_figure(test_batch, summary_writer, classes, val_targets_,
                                             val_predictions_, global_step=global_step)

                df = pd.DataFrame(stats)
                savemsg = ""
                if len(df) > 2:

                    previous = df.iloc[1:-1]
                    current = df.iloc[-1]

                    # check if model improved
                    if (current.valloss <= previous.valloss).all():
                        savemsg = "saving model"
                        torch.save(model.state_dict(), os.path.join(args.output_folder, f"model_best.pth"))
                        torch.save(meta_optimizer.state_dict(), os.path.join(args.output_folder, f"optimizer_best.pth"))

                        # Test on Sen12MS testset
                        test_predictions, test_targets = [], []
                        for _ in range(args.n_testtasks // args.batch_size):
                            test_idx, test_batch = next(testdataloader_enumerate)
                            valloss_, val_predictions_, val_targets_, val_id_ = meta_test_episode(test_batch, model,
                                                                                                  criterion, args)
                            test_predictions.append(val_predictions_)
                            test_targets.append(val_targets_)
                        testaccuracy = float((torch.hstack(test_predictions).cpu().detach().numpy() == torch.hstack(test_targets).cpu().detach().numpy()).astype(float).mean())
                        with open(os.path.join(args.output_folder, "sen12ms_testaccuracy.txt"), "w") as f:
                            f.write(str(testaccuracy))
                        savemsg += f" testaccuracy: {testaccuracy:.2f}"

                        not_improved_counter = 0
                    else: # not improved
                        savemsg = f"not improved on previous best model (valloss={previous.valloss.min():.2f}) for {not_improved_counter+1}/{args.patience} log cycles until early stopping"
                        if not_improved_counter > args.patience:
                            print(f"model did not improve best model with after {args.patience} log cycles ({args.patience*args.log_every_n_batches} episodes)")
                            print(f"stopping training")
                            break
                        not_improved_counter += 1

                print()
                print(f"episode {batch_idx}: loss meta-train|meta-test={stats[-1]['trainloss']:.2f}|{stats[-1]['valloss']:.2f}, "
                      f"meta-val acc={stats[-1]['valacc']:.2f}. {savemsg}")
                df.to_csv(os.path.join(args.output_folder, f"log.csv"))

            if batch_idx >= args.num_batches:
                break


def meta_train_episode(batch, model, meta_optimizer, criterion, args):
    model.zero_grad()

    step = step_maml

    train_inputs, train_targets, train_id = batch['train']
    train_inputs = train_inputs.to(device=args.device).float()
    train_targets = train_targets.to(device=args.device)

    test_inputs, test_targets, test_id = batch['test']
    test_inputs = test_inputs.to(device=args.device).float()
    test_targets = test_targets.to(device=args.device)

    if args.not_one_vs_all:
        train_targets = reset_indices(train_targets)
        test_targets = reset_indices(test_targets)
    else:
        idx = torch.randint(args.num_ways, (1,)).to(args.device)  # take one of way classes
        train_targets = (reset_indices(train_targets) == idx).float()
        test_targets = (reset_indices(test_targets) == idx).float()


    outer_loss = torch.tensor(0., device=args.device)
    accuracy = torch.tensor(0., device=args.device)
    test_predictions_list = list()
    test_targets_list = list()
    for task_idx, (train_input, train_target, test_input,
                   test_target) in enumerate(zip(train_inputs, train_targets,
                                                 test_inputs, test_targets)):

        outer_loss_, predictions, test_target = step(model, train_input, train_target, test_input, test_target, criterion, args)

        accuracy += (predictions.view(-1).cpu() == test_target.view(-1).cpu()).float().mean()
        outer_loss += outer_loss_

        test_predictions_list.append(predictions)
        test_targets_list.append(test_target)

    outer_loss.div_(args.batch_size)
    accuracy.div_(args.batch_size)

    outer_loss.backward()
    meta_optimizer.step()

    if bool(torch.isnan(outer_loss)):
        raise ValueError("train outer-loss became nan!")

    return outer_loss, torch.cat(test_predictions_list), torch.cat(test_targets_list), np.array(test_id).T.reshape(-1)

def step_maml(model, train_input, train_target, test_input, test_target, criterion, args):
    model.zero_grad()

    if isinstance(model, torch.nn.DataParallel):
        meta_named_parameters = model.module.meta_named_parameters()
    else:
        meta_named_parameters = model.meta_named_parameters()

    params = OrderedDict(meta_named_parameters)

    for t in range(args.gradient_steps):

        train_logit = model(train_input, params=params)
        inner_loss = criterion(train_logit.squeeze(1), train_target)
        params = update_parameters(model=model, loss=inner_loss, params=params,
                                   inner_step_size=args.inner_step_size, first_order=args.first_order)

    test_logit = model(test_input, params=params)

    outer_loss = criterion(test_logit.squeeze(1), test_target)

    if args.not_one_vs_all:
        predictions = test_logit.argmax(1).cpu().detach()
    else:
        predictions = (torch.sigmoid(test_logit.squeeze(1)) > 0.5).cpu().detach()

    return outer_loss, predictions, test_target

def meta_test_episode(batch, model, criterion, args):
    model.zero_grad()

    step = step_maml

    train_inputs, train_targets, train_id = batch['train']
    train_inputs = train_inputs.to(args.device).float()
    train_targets = train_targets.to(args.device)

    test_inputs, test_targets, test_id = batch['test']
    test_inputs = test_inputs.to(args.device).float()
    test_targets = test_targets.to(args.device)

    if args.not_one_vs_all:
        train_targets = reset_indices(train_targets)
        test_targets = reset_indices(test_targets)
    else: # default one vs all
        idx = torch.randint(args.num_ways, (1,)).to(args.device)  # take one of way classes
        train_targets = (reset_indices(train_targets) == idx).float()
        test_targets = (reset_indices(test_targets) == idx).float()

    batch_size = test_inputs.shape[0]
    outer_loss = torch.tensor(0., device=args.device)

    test_predictions_list = list()
    test_targets_list = list()
    for task_idx, (train_input, train_target, test_input,
                   test_target) in enumerate(zip(train_inputs, train_targets,
                                                 test_inputs, test_targets)):

        outer_loss_, predictions, test_target = step(model, train_input, train_target, test_input, test_target, criterion, args)
        outer_loss += outer_loss_

        test_predictions_list.append(predictions)
        test_targets_list.append(test_target.cpu().detach())

    outer_loss.div_(batch_size)

    if bool(torch.isnan(outer_loss)):
        raise ValueError("test outer-loss became nan!")

    return outer_loss, torch.cat(test_predictions_list), torch.cat(test_targets_list), np.array(test_id).T.reshape(-1)

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())