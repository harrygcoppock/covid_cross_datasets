import sys
import os
import glob
import re
from itertools import product

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from model import Conv_Model
from data_loader import COVID_dataset
from utils.confusion_matrix import plot_confusion

import warnings
warnings.filterwarnings("ignore")

import wandb

def make_sample_weights(data_set, args):
    '''
    makes the weights for each of the classes
    '''
    data_set_metadata = data_set.metadata
    num_pos = data_set_metadata[1].value_counts()[1]
    num_neg = data_set_metadata[1].value_counts()[0]
    pos_weight = num_neg/num_pos
    print('weight for covid', pos_weight)
    weights = torch.Tensor([pos_weight])

    return weights

def run_train(epoch, loader_train, model, device, optimizer, weight, args):
    model.train()
    loader_train = tqdm(loader_train, position=0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    for i, (audio, label) in enumerate(loader_train):
        model.zero_grad()

        audio = audio.to(device)
        label = label.to(device)
        predicts_soft = model(audio)

        loss = criterion(predicts_soft, label.unsqueeze(1).float())
        loss.backward()

        # get accuracy
        predicts = torch.sigmoid(predicts_soft.detach())
        predicts = np.where(predicts.cpu().numpy()>0.5, 1, 0)
        score = f1_score(label.cpu().numpy(), predicts)

        # if scheduler is not None:
        #     scheduler.step()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        loader_train.set_description(
            (f'epoch: {epoch + 1}; f1: {score:.5f}; loss: {loss.item():.3f}'))

        if args.logger == 'wandb':
            wandb.log({"F1": score, "loss": loss.item()})

        
    if args.fold_id == 'all' and epoch % 10 == 0:
        '''
        when training on 'all' simply save the model every 10 epochs and we can evaluate them individually
        '''
        new_dirname = os.path.join(args.dirname, f"{epoch}")
        os.mkdir(new_dirname)
        save_model(model, new_dirname)
        print(F"saving model {new_dirname}")
        return


def eval(model, audio, label, device, criterion):

    audio = audio.to(device)
    label = label.to(device)
    predicts_soft = model(audio)

    loss = criterion(predicts_soft, label.unsqueeze(1).float())
    # get accuracy
    predicts_soft = torch.sigmoid(predicts_soft).cpu().numpy()
    predicts = np.where(predicts_soft > 0.5, 1, 0)
    return loss, predicts, predicts_soft


def run_eval(epoch, loader_test, model, device, weight, args, do_save_model=True):
    model.eval()
    loader_test = tqdm(loader_test, position=0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    with torch.no_grad():
        y_hts = []
        ys = []
        logits_list = []
        losses = []
        for i, (audio, label) in enumerate(loader_test):
            label = label.to(device)
            if args.eval_type != 'maj_vote':
                loss, predicts, predicts_soft = eval(model, audio, label, device, criterion)
            else:
                clips = audio
                clip_loss, clip_predicts = 0, []
                for audio in clips:
                    loss, predicts, predicts_soft = eval(model, audio, label, device, criterion)
                    clip_loss += loss
                    clip_predicts.append((predicts, predicts_soft))

                # Aggregate predicts and loss
                loss = clip_loss / len(clips)
                positive = np.count_nonzero([c[0] for c in clip_predicts])
                votes = {'1': positive, '0': len(clip_predicts)-positive}
                # If its a tie, use logits
                if votes['1'] == votes['0']:
                    logits = (
                        sum([c[1] for c in clip_predicts if c[0].item() == 0]), # Negative
                        sum([c[1] for c in clip_predicts if c[0].item() == 1]), # Positive
                    )
                    predicts = np.argmax(logits).reshape(1,1)
                else:
                    predicts = np.array(int(max(votes.items(), key=lambda x: x[1])[0])).reshape(1,1)

            # for ROC-AUC
            if args.eval_type != 'maj_vote':
                logits_list.extend(average.squeeze().tolist())
            else:
                average_logits = [c[1][0][0] for c in clip_predicts]
                if args.max_logit:
                    logits_list.append(np.max(average_logits)) 
                else:
                    logits_list.append(np.mean(average_logits)) 

            y_hts.append(predicts)
            ys.append(label.cpu().numpy())
            losses.append(loss.item())

        ys = np.concatenate(ys)
        y_hts = np.concatenate(y_hts)

        fig = plot_confusion(ys, y_hts)

        score = f1_score(ys, y_hts)
        acc = accuracy_score(ys, y_hts)
        recall = recall_score(ys, y_hts, average='macro')
        # ROC-AUC
        fpr, tpr, _ = roc_curve(ys, logits_list)
        roc_auc = auc(fpr, tpr)

        if args.do_test: ##alican: look here lastly
            path = os.path.join(args.saved_model_dir, 'test_results_' + str(roc_auc) + '.txt')
            dir_words = args.saved_model_dir.split("/")
            txt_name = dir_words[3] + '_' + dir_words[-2] + '_' + dir_words[-1] + '_testAuc_' + str(roc_auc) + '.txt'
            # I do not have permission to save to alicans's dir
            if 'hgc19' in path and 'aa9120' in os.getcwd():
                path = os.path.join('/vol/bitbucket/aa9120/Dicova_Imperial/models',  txt_name)
                with open(path, 'w+') as f:
                    for file_name, logit in zip(loader_test.iterable.dataset.train_fold, logits_list):
                        f.write(f'{file_name} {logit}\n')
                    if args.modality == "speechbreath":
                        f.write(f'test roc_auc is: {roc_auc}\n')
            else:
                with open(path, 'w+') as f:
                    for file_name, logit in zip(loader_test.iterable.dataset.train_fold, logits_list):
                        f.write(f'{file_name} {logit}\n')

                    if args.modality == "speechbreath":
                        f.write(f'test roc_auc is: {roc_auc}\n')

            return

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        #plt.show()

        print(roc_auc)
        loader_test.set_description((f'epoch: {epoch + 1}; test f1: {score:.5f}; test AUC {roc_auc:.5f}'
                                        f'test loss: {sum(losses)/len(losses):.3f}'))

        outputs = {"Test F1": score,
                    "Test loss": sum(losses) / len(losses),
                    "epoch": epoch, "Test acc": acc,
                    "test AUC": roc_auc,
                    "ROC": wandb.Image(plt),
                    "Confusion": wandb.Image(fig),
                    "UAR": recall,
                    "FPR": fpr,
                    "TPR": tpr}
        # print(outputs)

        if args.logger == 'wandb':
            wandb.log(outputs)

        # Save model if required
        if args.dirname and args.do_train and do_save_model and epoch >= 20:
            roc_auc *= 100
            prevs = glob.glob(args.dirname+'/*F1*') + glob.glob(args.dirname+'/*AUC*')
            new_dirname = os.path.join(args.dirname, f"{args.task}_AUC-{roc_auc:.3f}")
            # Overwrite current worst if already k saved models and current model
            # better than an existing one
            if len(prevs) == args.save_model_topk:
                func = lambda p: float(re.search(
                    r"(?<=(?:F1)-)[0-9]+\.[0-9]{3}$" if 'F1' in p else r"(?<=(?:AUC)-)[0-9]+\.[0-9]{3}$",
                    p).group(0))
                scores = [func(p) for p in prevs]
                if roc_auc not in scores and roc_auc > min([func(p) for p in prevs]):
                    replace_dir = sorted(prevs, key=func)[0]
                    # issues with multiple threads and new_dirname already existing
                    if not os.path.exists(new_dirname):
                        os.rename(replace_dir, new_dirname)
                    save_model(model, new_dirname)
            else:
                if not os.path.exists(new_dirname):
                    os.mkdir(new_dirname)
                    save_model(model, new_dirname)
                    with open(os.path.join(new_dirname, 'epoch.txt'), 'w') as f:
                        f.write(str(epoch + 1))


def save_model(model, new_dirname):
    path = os.path.join(new_dirname, 'model.pt')
    torch.save(model.state_dict(), path)
    with open(os.path.join(new_dirname, 'config.txt'), 'w') as f:
        f.write(str(model))


def main(args):
    ''' Launch training and eval.
    '''
    hyp_params = dict(
        dropout=False,
        depth_scale=args.depth_scale,
        lr=args.lr,
        wsz=args.wsz,
        batch_size=args.batch_size,
        n_fft=args.nfft,
        sample_rate=args.sr,
        task=args.task,
        modality=args.modality,
        n_mfcc=args.n_mfcc,
        onset_sample_method = args.onset_sample_method,
        kdd = args.kdd,
        feature_type=args.feature_type,
        repetitive_padding = args.repetitive_padding,
        scheduler=args.scheduler,
        max_logit=args.max_logit,
        compare_dataset=args.compare_dataset

    )
    # Init dir for saving models
    args.dirname = None
    if args.save_model_topk > 0 and not args.do_test:
        existing = glob.glob('models/'+str(args.dataset)+'/run_*')
        idxs = [int(re.search(r"(?<=_)[0-9]+$", s).group(0)) for s in existing]
        ext = max(idxs) + 1 if idxs else 1
        args.dirname = os.path.join(os.getcwd(), 'models', str(args.dataset), 'run_'+str(ext))
        assert not os.path.exists(args.dirname), f"Dirname already exists:\nf{dirname}"
        os.mkdir(args.dirname)
        with open(os.path.join(args.dirname,'args.txt'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    if args.logger == 'wandb' and not args.do_test:
        run = wandb.init(project='cross_datasets'+args.dataset,
                         reinit=True,
                         config=hyp_params)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise() if args.noise else NoneTransform(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Train and dev split
    device = 'cuda'

    print(args)
    if not args.do_test:
        train_dataset = COVID_dataset(
            dset='train',
            transform=transform_train,
            window_size=hyp_params["wsz"],
            n_fft=args.nfft,
            sample_rate=args.sr,
            masking=args.masking,
            pitch_shift=args.pitch_shift,
            modality=args.modality,
            kdd=args.kdd,
            feature_type=args.feature_type,
            n_mfcc = args.n_mfcc,
            onset_sample_method = args.onset_sample_method,
            repetitive_padding=args.repetitive_padding,
            dataset=args.dataset
        )


        val_dataset = COVID_dataset(
            dset='val',
            transform=transform_test,
            eval_type=args.eval_type,
            window_size=hyp_params["wsz"],
            n_fft=args.nfft,
            sample_rate=args.sr,
            modality=args.modality,
            feature_type=args.feature_type,
            n_mfcc = args.n_mfcc,
            onset_sample_method = args.onset_sample_method,
            repetitive_padding=args.repetitive_padding,
            dataset=args.dataset
        )

        print('length of training dataset', len(train_dataset.train_fold))
        print('length of validation dataset', len(val_dataset.train_fold))

        batch_size = args.batch_size
        train_weight = make_sample_weights(train_dataset, args).to(device)
        val_weight = make_sample_weights(val_dataset, args).to(device)

        loader_train = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)
        

        loader_dev = DataLoader(val_dataset,
                                batch_size=batch_size if args.eval_type != 'maj_vote' else 1,
                                shuffle=True,
                                num_workers=4)


    # Model
    if args.feature_type == 'stft':
        input_shape = (int(1024*args.nfft/2048)+1,int(94*args.wsz*args.sr/48000))
    if args.feature_type == 'mfcc':
        input_shape = (args.n_mfcc, int(94 * args.wsz * args.sr / 48000))
        
    model = Conv_Model(
        dropout=args.dropout,
        depth_scale=args.depth_scale,
        input_shape=input_shape,       # Dynamically adjusts for different input sizes
        device=device,
        modality=args.modality
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))


    if args.scheduler:
        print('using scheduler')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                10, 
                                                                eta_min=0.00001, 
                                                                last_epoch=-1, 
                                                                verbose=True)
    # Load saved model
    if args.saved_model_dir and args.do_test:
        path = os.path.join(args.saved_model_dir, 'model.pt')
        model.load_state_dict(torch.load(path))
        print('Loading saved model:', args.saved_model_dir)

    if args.max_logit:
        assert args.eval_type == 'maj_vote', 'taking the max of the votes only works if maj voting'
    if args.logger == 'wandb' and not args.do_test:
        wandb.watch(model)
    if args.do_train:
        for epoch in range(100):
            run_train(epoch, loader_train, model, device, optimizer, train_weight, args)
            if epoch % args.n_epoch_val == 0:
                run_eval(epoch, loader_dev, model, device, val_weight, args)
            if args.scheduler:
                if epoch % scheduler.T_max == 0:
                    print('Reset Scheduler')
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                scheduler.T_max, 
                                                                eta_min=scheduler.eta_min, 
                                                                last_epoch=-1, 
                                                                verbose=True)

                scheduler.step()
    if args.do_test:

        test_dataset = COVID_dataset(
            dset='test',
            fold_id=None,
            transform=transform_test,
            eval_type=args.eval_type,
            window_size=hyp_params["wsz"],
            n_fft=args.nfft,
            sample_rate=args.sr,
            modality=args.modality,
            feature_type=args.feature_type,
            n_mfcc = args.n_mfcc,
            onset_sample_method = args.onset_sample_method,
            repetitive_padding=args.repetitive_padding,
            dataset=args.dataset
        )
        loader_test = DataLoader(test_dataset,
                        batch_size=args.batch_size if args.eval_type != 'maj_vote' else 1,
                        shuffle=False,
                        num_workers=4)
        run_eval(None, loader_test, model, device, None, args)



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=5.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if self.std == 0.:
            return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std)


class NoneTransform(object):
    """ Does nothing to the tensor, to be used instead of None
    
    Args:
        data in, data out, nothing is done
    """
    def __call__(self, x):
        return x


def parse_args():
    parser = argparse.ArgumentParser(description='COVID_detector')
    # Hparam args
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.0001)
    parser.add_argument('--dropout', type=bool, help='Drop out if true it is fixed at 0.5. Note for now it only applies to the first layer', default=False)
    parser.add_argument('--depth_scale', type=float, help='a parameter which multiplies the number of channels.', choices=[1, 0.5, 0.25, 0.125], default=1)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--wsz', type=int, help='Size of the audio clip window size (seconds).', default=1)
    parser.add_argument('--nfft', type=int, help='n_fft parameter', default=2048)
    parser.add_argument('--sr', type=int, help='sample rate parameter', default=48000)
    parser.add_argument('--n_mfcc', type=int, help='number of mfcc components', default=20)
    parser.add_argument('--feature_type', type=str, help='which feature type you want to use to preprocess sound data? (mfcc, stft)', default='stft')
    parser.add_argument('--onset_sample_method', type=bool, help='do you want to select cough peak samples rather than rand index', default=False)
    parser.add_argument('--repetitive_padding', type=bool, help='enables padding with signal repetition (rather than 0 padding)', default=False)
    # augmentation args
    parser.add_argument('--masking', type=bool, help='do we perform time and frequency masking or not?', default=False)
    parser.add_argument('--pitch_shift',type=bool,help='perform a pitch shift provides the step size ot shift',default=False)
    parser.add_argument('--noise', type=bool, help='add gaussian noise to the specgram', default=False)
    # Config arg
    parser.add_argument('--logger', type=str, help='Type of logger to use.', choices=['default', 'wandb'], default='wandb')
    parser.add_argument('--save_model_topk', type=int, help='Save the k best performing models according to validation F1.', default=3)
    parser.add_argument('--do_train', type=bool, help='Run training loop.', default=True)
    parser.add_argument('--do_test', type=bool, help='Produce logits for test set?', default=True)
    parser.add_argument('--eval_type', type=str, help='Type of eval to run', choices=['beginning', 'random','maj_vote'], default='maj_vote')
    parser.add_argument('--saved_model_dir', type=str, help='Path to dir containing saved model.', default="")
    parser.add_argument('--scheduler', type=bool, default=False)
    parser.add_argument('--max_logit', type=bool, help='when majority voting do you take the mean or the max logit for ROC-AUC?')
    parser.add_argument('--n_epoch_val', type=int, help='determine the validation run for every n epochs', default=1)
    # What task to do
    parser.add_argument('--location', type=str, help='where is the data', default='bitbucket', choices=['bitbucket', 'hpc'])
    parser.add_argument('--modality', type=str, help='do we want to stack all the modalities together, if so how many are there?', default='cough')
    parser.add_argument('--compare_dataset', type=bool, help='do you want to train on compare dataset?', default=False)

    # Hack to use script with debugger by loading args from file
    if len(sys.argv) == 1:
        print('using txt')
        #alican
        with open(os.getcwd()+'/args.txt', 'r') as f:
            args = argparse.Namespace(**json.loads(f.read()))
    else:
        print('not using txt')
        args = parser.parse_args()


    print(args)
    if args.eval_type == 'maj_vote' and args.batch_size > 1:
        print(f"If running majority voting eval, eval batch size must be 1.")

    # Other processing
    assert not (args.do_train and args.saved_model_dir), f"Can't run training and load saved model. Investigate!"
    if args.saved_model_dir:
        saved_model_args_path = os.path.join(os.path.dirname(args.saved_model_dir), 'args.txt')
        with open(saved_model_args_path, 'r') as f:
            saved_model_args = json.loads(f.read())
        # Combine current args with saved model params
        hparams = ["lr", "dropout", "depth_scale", "wsz", "nfft", "sr",
                     "feature_type", "n_mfcc", "onset_sample_method",
                    "repetitive_padding", "scheduler",
                    "max_logit"]
        for k,v in saved_model_args.items():
            if k in hparams:
                setattr(args, k, v)
            elif not hasattr(args, k):
                setattr(args, k, None)
        print('using these saved model ags:', args)



    return args


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    
    # for the covid cross cultures paper we need to train on one dataset
    # then choose the best model based on val set but then evaluate on the 
    # other datasets test set!

    main(args)


