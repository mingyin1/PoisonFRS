import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--attack', nargs='?', default='Our', help="Specify a attack method")
    parser.add_argument('--dim', type=int, default=32, help='Dim of latent vectors.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--k', type=int, default=5, help='k popular items in our attack.')
    parser.add_argument('--Lambda', type=float, default=10, help='Amptitude of malicious model weights.')
    parser.add_argument('--alpha', type=float, default=1, help='Strength of dragging model updates to be malicious.')
    parser.add_argument('--attack_round', type=int, default=50, help='Start to attack after several global rounds.')
    parser.add_argument('--dataset', nargs='?', help='Choose a dataset.')
    parser.add_argument('--device', nargs='?', default='cuda',
                        help='Which device to run the model.')

    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4000, help='Batch size.')

    parser.add_argument('--grad_limit', type=float, default=3., help='Limit of l2-norm of item gradients.')
    parser.add_argument('--clients_limit', type=float, default=0.05, help='Limit of proportion of malicious clients.')
    parser.add_argument('--items_limit', type=int, default=60, help='Limit of number of non-zero item gradients.')
    parser.add_argument('--part_percent', type=int, default=0.01, help='Proportion of attacker\'s prior knowledge.')

    parser.add_argument('--attack_lr', type=float, default=0.001, help='Learning rate on FedRecAttack.')
    parser.add_argument('--attack_batch_size', type=int, default=256, help='Batch size on FedRecAttack.')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'median', 'trim', 'clip', 'krum', 'flame','hics'], help='Aggregation rule.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU.')
    parser.add_argument('--noisy', action='store_true', help='Add noise to lambda.')
    parser.add_argument('--adapt', action='store_true', help='Adaptively adjust alpha.')

    return parser.parse_args()


args = parse_args()
