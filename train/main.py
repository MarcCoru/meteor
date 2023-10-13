import argparse
import json
import os

from evaluate_dfc2020 import main as eval_dfc
from maml import train as maml_train

def parse_args():
    parser = argparse.ArgumentParser('A central script to select learning scheme and mode')

    # general arguments
    parser.add_argument('--dataset-path', type=str, default="/data/sen12ms",  # /data2/sen12ms128
                        help='path to sen12ms dataset. requires sen12ms.h5 and sen12ms.csv files')
    parser.add_argument('--dfc-path', type=str, default="/data/sen12ms/DFC_Public_Dataset",  # /data2/sen12ms128
                        help='path to sen12ms dataset. requires sen12ms.h5 and sen12ms.csv files')
    parser.add_argument('--output-folder', type=str, default="/tmp",  #
                        help='Path to the output folder for saving the model.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs. Only applicable with SGD learning.')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading (default: 0).')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA if available.')
    parser.add_argument('--tensorboard-log-images', action='store_true',
                        help='additionally logs images to tensorboard.')
    parser.add_argument('--snapshot', action='store_true',
                        help='additionally create snapshot on every log.')
    parser.add_argument('--reset_indices', action='store_true',
                        help='let model train with switching classes.')
    parser.add_argument('--not-one-vs-all', action='store_true',
                        help='performs one vs all classification.')
    parser.add_argument('--start-from-pretrained', action='store_true',
                        help='downloads the pretrained checkpoint and starts training from there.')

    # fixed parameter
    parser.add_argument('--num-shots', type=int, default=2,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=4,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden dimensions in the neural network model. This parameter controls model capactity.')
    parser.add_argument('--gradient-steps', type=int, default=1,
                        help='number of inner gradient steps.')
    parser.add_argument('--inner-step-size', type=float, default=0.1,
                        help='number of inner step size.')
    parser.add_argument('--outer-step-size', type=float, default=1e-4,
                        help='number of outer step size.')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay')

    parser.add_argument('--validate-after-n-episodes', type=int, default=200,
                        help='log and snapshot training state every n training episodes')

    parser.add_argument('--norm', type=str, default="instancenorm",  #
                        help='normalization of the resnet model. naming following Bronskill et al., 2020.')

    parser.add_argument('--s2only', action='store_true',
                        help='use only the 13 bands of Sentinel 2.')
    parser.add_argument('--dynamicresnet', action='store_true',
                        help='uses a dynamic band encoder with a variable number of labels.')
    parser.add_argument('--rgbonly', action='store_true',
                        help='use only the 3 RGB bands of Sentinel 2.')

    parser.add_argument('--learn-inner-learning-rates', action='store_true',
                        help='use the maml++ feature of learning inner learning rates.')
    parser.add_argument('--first-order', action='store_true',
                        help='perform first order updates.')

    parser.add_argument('--resnet', action='store_true',
                        help='use a resnet12.')

    # early stopping
    parser.add_argument('--log-every-n-batches', type=int, default=100,
                        help='log and snapshot training state every n training episodes')
    parser.add_argument('--patience', type=int, default=20,
                        help='number of log cycles without improvement until early stopping')

    parser.add_argument('--n_testtasks', type=int, default=1000,
                        help='number of testbatches on sen12ms')

    # sparseMAML
    parser.add_argument('--gradient-mask', action='store_true',
                        help='Use CUDA if available.')
    parser.add_argument('--prototypicalnetwork', action='store_true',
                        help='Use CUDA if available.')

    return parser.parse_args()
def main():
    args = parse_args()

    args.device = 'cuda' if args.use_cuda else 'cpu'

    print(f"storing results in {args.output_folder}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(os.path.join(args.output_folder,"args.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    maml_train(args)

    if args.one_vs_all:
        # after training evaluate on the DFC Dataset
        args.ensemble = False
        args.first_order = True
        args.gradient_steps = 60
        for shots in [1,2,5,10,15]:
            args.num_shots = shots
            eval_dfc(args)

if __name__ == '__main__':
    main()