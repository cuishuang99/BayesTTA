import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from scipy.stats import f
import torch
import torch.nn.functional as F
from torch.linalg import slogdet
import operator
import torch.nn as nn
import clip
from utils import *
import csv
from scipy.stats import chi2
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET

def load_partial_args_from_xml(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_args = {}
    for tag in ['model_ema_beta', 'learning_rate', 'alpha']:
        if root.find(tag) is not None:
            xml_args[tag.replace('-', '_')] = float(root.find(tag).text)
    return xml_args


def get_arguments():
    """Get arguments of the test-time adaptation with hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to XML config file')

    # Required arguments
    parser.add_argument('--datasets', required=True, help="Datasets to process, separated by '/'.")
    parser.add_argument('--backbone', required=True, choices=['RN50', 'ViT-B/16'], help='CLIP model backbone.')

    # Tunable hyperparameters
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for GMM adjustment term.')
    parser.add_argument('--model-ema-beta', type=float, default=0.99, help='EMA decay rate for model parameters.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for LN or BN layers.')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Covariance regularization strength.')
    parser.add_argument('--sigma', type=float, default=0.1, help='Prior covariance scale.')
    parser.add_argument('--batch-size', type=int, default=128, help='Test batch size.')

    # Optional arguments
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging.')
    parser.add_argument('--data-root', default='/home/csh/dataset/MM_Dataset/', help='Path to datasets.')
    parser.add_argument('--output', default='./output/', help='Output directory.')

    args = parser.parse_args()

    if args.config:
        xml_args = load_partial_args_from_xml(args.config)
        for key, val in xml_args.items():
            setattr(args, key, val)  # 直接覆盖命令行参数

    return args

def print_hyperparameters(args):
    """Print all tunable hyperparameters"""
    print("\n=== Hyperparameter Settings ===")
    print(f"GMM Adjustment Alpha: {args.alpha}")
    print(f"Covariance Regularization Epsilon: {args.epsilon}")
    print(f"Prior Covariance Sigma: {args.sigma}")
    print(f"Model EMA Beta: {args.model_ema_beta}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Test Batch Size: {args.batch_size}")
    print("===============================\n")

def initialize_state(clip_weights, args):
    """Initialize GMM state using command-line arguments"""
    K, D = clip_weights.shape[1], clip_weights.shape[0]
    return {
        'mu': clip_weights.T.clone().cuda(),
        'sigma_sums': torch.zeros(K, D, D).cuda(),
        'sigma_sum': torch.zeros(D, D).cuda(),
        'sigma_counts': torch.zeros(K).cuda(),
        'sigma_count': torch.tensor(0.0).cuda(),
        'ct': torch.ones(K).cuda(),
        '_initialized': False,
        'D': D,
        'K': K,
        'cfg': {
            'epsilon': args.epsilon,
            'sigma': args.sigma,
            'alpha': args.alpha,
            'use_lda': True,  # Use LDA H0 hypothesis by default
        },
    }

def get_domains(dataset_name):
    """Generate domain parameter list based on dataset name"""
    domain_map = {
        'low_light_cifar10': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75],
        'fmow': list(range(16)),
        'yearbook': list(range(1930, 2014)),
        'rmnist': list(range(0, 90, 10))
    }
    return domain_map.get(dataset_name, [])

def compute_loss(logits_zs, logits_adapted):
    """Compute cross-entropy loss (supports soft labels)"""
    # Target distribution: softmax probability of logits_adapted (detach to prevent gradient flow)
    target_probs = F.softmax(logits_adapted.detach(), dim=1)  # q ∈ R^{B×K}
    
    # Cross-entropy loss: -Σ(q * log p)
    log_probs = F.log_softmax(logits_zs, dim=1)   # log p ∈ R^{B×K}
    loss = -torch.mean(torch.sum(target_probs * log_probs, dim=1))  # scalar
    
    return loss

def initialize_ema_params(clip_model, backbone):
    """Initialize EMA parameters for LN or BN layers depending on the backbone type"""
    ema_params = {}
    for name, param in clip_model.named_parameters():
        if (backbone == 'RN50' and 'bn' in name.lower()) or \
           (backbone == 'ViT-B/16' and 'ln' in name.lower()):
            # print(name)  # Output matched layer names for debugging
            ema_params[name] = param.data.clone().detach()
    return ema_params

def apply_ema(clip_model, ema_params, beta=0.99):
    """Apply EMA update to LN or BN layer parameters"""
    with torch.no_grad():
        for name, param in clip_model.named_parameters():
            if name in ema_params:
                ema_params[name] = beta * ema_params[name] + (1 - beta) * param.data
                param.data.copy_(ema_params[name])    

def get_ln_bn_params(clip_model, backbone):
    """Get parameters of LN or BN layers depending on the backbone type"""
    if backbone == 'RN50':
        ln_bn_params = [param for name, param in clip_model.named_parameters() if 'bn' in name.lower()]
    elif backbone == 'ViT-B/16':
        ln_bn_params = [param for name, param in clip_model.named_parameters() if 'ln' in name.lower()]
    return ln_bn_params                

def _covariance_homogeneity_test(x, P_zs, K, alpha=0.05):
    """Perform covariance homogeneity test based on the first batch and class probabilities"""
    # ================== Data preprocessing ==================
    device = x.device
    B, D_orig = x.shape
    
    # PCA dimensionality reduction to 10D (on CPU)
    pca = PCA(n_components=10)
    x_lowdim_np = pca.fit_transform(x.cpu().numpy())  # [B, 10]
    x_lowdim = torch.tensor(x_lowdim_np, dtype=torch.float32, device='cpu')  # Keep on CPU
    D = 10
    
    # ================== Covariance matrix computation ==================
    eps = 1e-8
    reg = 1e-6 * torch.eye(D, device='cpu')  # Regularization term (CPU tensor)
    
    covs, counts = [], []
    for k in range(K):
        pk = P_zs[:, k].cpu()  # [B]
        mu_k = torch.sum(pk[:, None] * x_lowdim, dim=0) / (torch.sum(pk) + eps)  # [D]
        diff = x_lowdim - mu_k  # [B, D]
        weighted_diff = torch.einsum('b,bd,be->de', pk, diff, diff)  # [D, D]
        cov_k = (weighted_diff + reg) / (torch.sum(pk) + eps)
        covs.append(cov_k)
        counts.append(torch.sum(pk))

    # ================== Box's M test ==================
    total_dof = sum([counts[k] - 1 for k in range(K)])

    if total_dof <= (K - 1):
        print("Insufficient degrees of freedom for Box's M test.")
        return False

    S_pooled = sum([(counts[k] - 1) / (total_dof + eps) * covs[k] for k in range(K)])

    M = 0.0
    valid_classes = 0
    for k in range(K):
        nk = counts[k]
        Sk = covs[k]

        # Use slogdet instead of logdet for better stability
        sign_Sp, logdet_Sp = torch.linalg.slogdet(S_pooled)
        sign_Sk, logdet_Sk = torch.linalg.slogdet(Sk)

        if sign_Sp <= 0 or sign_Sk <= 0:
            print("Non-positive definite covariance matrix encountered.")
            return False

        M += (nk - 1) * (logdet_Sp - logdet_Sk)
        valid_classes += 1

    if valid_classes < 2:
        return False

    # ================== F-distribution parameters ==================
    try:
        a_prime = 1 - (2*D**2 + 3*D - 1)/(6*(D+1)*(valid_classes-1)) * (
            sum(1/(counts[k]-1 + eps) for k in range(K) if counts[k] >= D+1) - 1/total_dof
        )
        denominator = total_dof - valid_classes
        if denominator <= 0:
            print("Denominator for a_dprime is non-positive.")
            return False
        a_dprime = (D*(D+1)*(valid_classes-1)*(valid_classes+1)) / (6 * denominator)

        d1 = D * (D + 1) * (valid_classes - 1) // 2
        d2 = (d1 + 2) / (a_dprime + 1e-8)
        M_corrected = M * a_prime
        F_stat = M_corrected / (d1 * (1 - a_prime + M_corrected / (d2 + eps)))
        F_critical = f.ppf(1 - alpha, d1, d2)
        reject_h0 = F_stat > F_critical

        print(f"F_stat: {F_stat:.4f}, F_critical ({d1}, {d2:.2f}): {F_critical:.4f}")
        print(f"Reject H0: {reject_h0}")
        return reject_h0

    except Exception as e:
        print(f"[Error] Exception during F-stat calculation: {e}")
        return False

def test_time_adaptation_em(images, clip_model, clip_weights, state_dict):
    """Stable version supporting LDA/QDA"""
    device = images.device

    # Unpack state parameters (automatically handle dimensions based on mode)
    mu_prev = state_dict['mu']  # [K,D]
    D = state_dict['D']
    K = state_dict['K']
    cfg = state_dict['cfg']

    # Trigger covariance homogeneity test on the first run
    if not state_dict.get('_initialized', False):
        with torch.no_grad():
            x = clip_model.encode_image(images).float().squeeze(1)
            x = x / x.norm(dim=-1, keepdim=True)
            logits_zs = 100.0 * x @ clip_weights.float()
            P_zs = F.softmax(logits_zs, dim=1)

            # Perform the test and set the mode
            reject_h0 = _covariance_homogeneity_test(x, P_zs, K)
            cfg['use_lda'] = not reject_h0  # Use QDA if H0 is rejected, otherwise LDA
            print("Use Shared Covariance (LDA) cfg['use_lda']", cfg['use_lda'])
            state_dict['_initialized'] = True  # Mark as initialized

    # Handle dimensions for covariance-related parameters
    if cfg['use_lda']:
        # LDA mode parameters
        sigma_sum_prev = state_dict['sigma_sum']  # [D,D]
        sigma_count_prev = state_dict['sigma_count']  # scalar
    else:
        # QDA mode parameters
        sigma_sums_prev = state_dict['sigma_sums']  # [K,D,D]
        sigma_counts_prev = state_dict['sigma_counts']  # [K]

    with torch.no_grad():
        # ================== Feature Extraction and E-Step ==================
        x = clip_model.encode_image(images).float().squeeze(1)
        x = x / x.norm(dim=-1, keepdim=True)
        logits_zs = 100.0 * x @ clip_weights.float()
        P_zs = F.softmax(logits_zs, dim=1)

        # ================== M-Step Parameter Update ==================
        sum_pk = P_zs.sum(0)  # [K]
        sum_pk_x = torch.einsum('bk,bd->kd', P_zs, x)  # [K,D]

        # Mean update (shared for both modes)
        ct_new = state_dict['ct'] + sum_pk
        mu_new = (state_dict['ct'][:, None] * mu_prev + sum_pk_x) / (ct_new[:, None] + 1e-8)

        # ================== Covariance Update (mode-specific) ==================
        diff = x.unsqueeze(1) - mu_prev.unsqueeze(0)  # [B,K,D]

        if cfg['use_lda']:
            # LDA mode: global shared covariance
            weighted_diff = torch.einsum('bk,bkd,bke->de', P_zs, diff, diff)
            sigma_sum_new = sigma_sum_prev + weighted_diff
            sigma_count_new = sigma_count_prev + x.size(0)

            # Regularization and broadcasting
            sigma_reg = (sigma_sum_new / sigma_count_new) * (1 - cfg['epsilon']) + \
                        cfg['epsilon'] * torch.eye(D, device=device) * (cfg['sigma']**2)  # [D,D]
            sigma_regs = sigma_reg.unsqueeze(0).repeat(K, 1, 1)  # [K,D,D]
        else:
            # QDA mode: class-specific covariance
            weighted_diff = torch.einsum('bk,bkd,bke->kde', P_zs, diff, diff)  # [K,D,D]
            sigma_sums_new = sigma_sums_prev + weighted_diff
            sigma_counts_new = sigma_counts_prev + sum_pk

            # Regularization
            sigma_regs = []
            for k in range(K):
                sigma_k = sigma_sums_new[k] / (sigma_counts_new[k] + 1e-8)
                reg = (1 - cfg['epsilon']) * sigma_k + cfg['epsilon'] * torch.eye(D, device=device) * (cfg['sigma']**2)
                sigma_regs.append(reg)
            sigma_regs = torch.stack(sigma_regs)  # [K,D,D]

        # ================== Discriminant Function (shared) ==================
        sigma_invs = torch.linalg.pinv(sigma_regs)
        diff_mu = x.unsqueeze(1) - mu_new.unsqueeze(0)  # [B,K,D]
        quad_term = -0.5 * torch.einsum('bkd,kde,bke->bk', diff_mu, sigma_invs, diff_mu)

        # Determinant computation (with numerical stability)
        log_dets = []
        for s in sigma_regs:
            try:
                L = torch.linalg.cholesky(s)
                log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            except:
                log_det = torch.logdet(s)
            log_dets.append(-0.5 * log_det)
        log_dets = torch.stack(log_dets)  # [K]

        log_prior = torch.log(torch.tensor(1.0 / K, device=device))
        logits_gmm = quad_term + log_dets + log_prior
        logits_adapted = logits_zs + cfg['alpha'] * logits_gmm

        # ================== Update State (mode-specific) ==================
        state_dict['mu'] = mu_new
        state_dict['ct'] = ct_new

        if cfg['use_lda']:
            state_dict['sigma_sum'] = sigma_sum_new
            state_dict['sigma_count'] = sigma_count_new
        else:
            state_dict['sigma_sums'] = sigma_sums_new
            state_dict['sigma_counts'] = sigma_counts_new

    return logits_adapted, state_dict

def adaptation_step(images, clip_model, clip_weights, state_dict, ema_params, optimizer, args):
    """Single adaptation step using EMA beta"""
    # Generate pseudo labels
    with torch.no_grad():
        logits_adapted, updated_state = test_time_adaptation_em(images, clip_model, clip_weights, state_dict)

    # Gradient update
    optimizer.zero_grad()
    x = clip_model.encode_image(images).float()
    x = x / x.norm(dim=-1, keepdim=True)
    logits_zs = 100.0 * x @ clip_weights.float()
    loss = compute_loss(logits_zs, logits_adapted)
    loss.backward()
    optimizer.step()

    # EMA update
    apply_ema(clip_model, ema_params, args.model_ema_beta)

    return logits_adapted, updated_state, ema_params

def main():
    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    args = get_arguments()
    print_hyperparameters(args)  # Print hyperparameters

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Run TTA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")

        # Initialize global state and statistics (accumulated across subdomains)
        global_state = None
        overall_correct = 0
        overall_samples = 0

        # Only optimize LN layer parameters
        ema_params = initialize_ema_params(clip_model, args.backbone)
        ln_bn_params = get_ln_bn_params(clip_model, args.backbone)
        optimizer = torch.optim.Adam(ln_bn_params, lr=args.learning_rate)  # Small learning rate with momentum

        # Generate output filename dynamically
        save_file = f"{args.output}{dataset_name}_alpha{args.alpha}_lr{args.learning_rate}_ema{args.model_ema_beta}_bn{args.batch_size}.csv"
        with open(save_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Domain', 'Accuracy'])

            for domain in get_domains(dataset_name):
                test_loader, classnames, template = build_test_data_loader(
                    dataset_name + str(domain), args.data_root, preprocess, args.batch_size
                )

                # Initialize state on first run
                if global_state is None:
                    clip_weights = clip_classifier(classnames, template, clip_model)
                    global_state = initialize_state(clip_weights, args)

                domain_correct = 0
                domain_samples = 0
                for images, targets in tqdm(test_loader):
                    images, targets = images.cuda(), targets.cuda()

                    # Perform batch-wise adaptation
                    logits, global_state, ema_params = adaptation_step(
                        images, clip_model, clip_weights, global_state, ema_params, optimizer, args
                    )

                    # Count correct predictions
                    preds = logits.argmax(dim=1)  # [batch_size]
                    if targets.dim() == 2 and targets.size(1) == 1:
                        targets = targets.squeeze(1)
                    correct = (preds == targets).sum().item()
                    domain_correct += correct
                    domain_samples += targets.size(0)

                domain_acc = domain_correct / domain_samples * 100
                writer.writerow([f"{dataset_name}_{domain}", domain_acc])
                print(f"{dataset_name}_{domain}")
                print(domain_acc)

                overall_correct += domain_correct
                overall_samples += domain_samples

            print("overall_samples")
            print("average", overall_correct / overall_samples)
            writer.writerow(["average", overall_correct / overall_samples])


if __name__ == "__main__":
    main()