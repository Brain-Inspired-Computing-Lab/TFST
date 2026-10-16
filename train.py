import argparse
import time
import yaml
import os
import logging
import warnings
import numpy as np
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.activation_based import functional
import torch
import torch.nn as nn
import torchinfo
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from timm.data import (
    resolve_data_config,
)
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
)
from timm.models.helpers import clean_state_dict
from timm.utils import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
import model

# DATASET
from data_utils import ADFTDLoader, APAVALoader, PTBLoader, PTBXLLoader

DATASET_LOADERS = {
    "adftd": ADFTDLoader,
    "apava": APAVALoader,
    "ptb": PTBLoader,
    "ptbxl": PTBXLLoader,
}


def build_dataset(args, flag):
    """Build the loader for ``args.dataset`` on one split (TRAIN/VAL/TEST)."""
    if args.dataset not in DATASET_LOADERS:
        raise ValueError(
            "Unsupported dataset: {}. Expected one of: {}.".format(
                args.dataset, ", ".join(sorted(DATASET_LOADERS))
            )
        )
    return DATASET_LOADERS[args.dataset](
        args=args,
        root_path=args.data_dir,
        flag=flag,
        downsize_factor=args.downsize_factor,
    )

torch.backends.cudnn.benchmark = True
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments (see conf/)",
)

parser = argparse.ArgumentParser(description="TFST Training")

# Dataset / Model parameters
parser.add_argument(
    "-data-dir",
    metavar="DIR",
    default="",
    help="path to dataset",
)
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="adftd",
    choices=sorted(DATASET_LOADERS),
    help="dataset to train on",
)
parser.add_argument(
    "--model",
    default="TFS",
    type=str,
    metavar="MODEL",
    help='name of model to train (default: "TFS")',
)
parser.add_argument(
    "--layer",
    default=4,
    type=int,
    help="",
)
parser.add_argument(
    "--in-channels",
    default=3,
    type=int,
    help="",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (model default if None)",
)
parser.add_argument(
    "--time-steps",
    type=int,
    default=4,
    metavar="N",
    help="",
)
parser.add_argument(
    "--num-heads",
    type=int,
    default=8,
    metavar="N",
    help="",
)
parser.add_argument(
    "--mlp-ratio",
    type=int,
    default=4,
    metavar="N",
    help="expansion ratio of the embedding dimension in the MLP block",
)
parser.add_argument(
    "--attention-type",
    default="MLSA",
    type=str,
    choices=["MLSA", "MRHA"],
    help="spiking attention mechanism used in each encoder block",
)
parser.add_argument(
    "--downsize-factor",
    type=int,
    default=1,
    metavar="N",
    help="downsampling factor for the model",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Path to checkpoint to resume from for inference (default: none)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="weight decay (default: 0.0001)"
)
parser.add_argument(
    "--sched",
    default="step",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=3,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--FLOPS",
    action="store_true",
    default=False,
    help="if True, compute FLOPs and SynOps using only first batch; if False, use entire loader_eval",
)

# MISC
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 1)",
)

_logger = logging.getLogger("train")
stream_handler = logging.StreamHandler()
format_str = "%(asctime)s %(levelname)s: %(message)s"
stream_handler.setFormatter(logging.Formatter(format_str))
_logger.addHandler(stream_handler)
_logger.propagate = False


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def calc_firing_rate_stats(spike_dict):
    """Calculate firing rate and non-zero rate statistics from spike tensors.
    
    Returns:
        firing_rates: Dict mapping layer names to firing rate values
        non_zero_rates: Dict mapping layer names to non-zero rate values
        tensor_sizes: Dict mapping layer names to total number of elements in tensor
    """
    firing_rates = {}
    non_zero_rates = {}
    tensor_sizes = {}
    
    for layer_name, spike_tensor in spike_dict.items():
        # spike_tensor shape: (T, B, ...) where T is time steps
        if spike_tensor.numel() == 0:
            continue
            
        # Store tensor size
        tensor_sizes[layer_name] = spike_tensor.numel()
        
        # Calculate non-zero rate: percentage of time steps with any spikes
        non_zero_per_timestep = (spike_tensor.reshape(spike_tensor.shape[0], -1) > 0).any(dim=1).float()
        non_zero_rates[layer_name] = non_zero_per_timestep.mean().item()
        
        # Calculate firing rate: mean spike value across all dimensions
        firing_rates[layer_name] = spike_tensor.mean().item()
    
    return firing_rates, non_zero_rates, tensor_sizes


def get_primary_metric(args, metrics):
    """Primary metric used for model selection."""
    return metrics.get("F1", 0), "F1"


def get_secondary_metric(args, metrics):
    """Tie-breaker used when the primary metric is equal."""
    return metrics.get("top1", 0), "top1"


def is_better_metric(new_metrics, best_metrics_dict, args):
    """Compare checkpoints on the primary metric, then the secondary one."""
    if best_metrics_dict is None:
        return True

    new_primary, _ = get_primary_metric(args, new_metrics)
    best_primary, _ = get_primary_metric(args, best_metrics_dict)

    if new_primary > best_primary:
        return True

    if abs(new_primary - best_primary) < 1e-6:  # float comparison tolerance
        new_secondary, _ = get_secondary_metric(args, new_metrics)
        best_secondary, _ = get_secondary_metric(args, best_metrics_dict)
        return new_secondary > best_secondary

    return False


def register_spike_hooks(model):
    """Register forward hooks to capture spike outputs from all modules."""
    spike_dict = {}
    handles = []
    
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.numel() > 0:
                # Store the most recent output (likely the spike output)
                spike_dict[name] = output.detach().cpu()
        return hook
    
    # Register hooks on all modules that might produce spikes
    # Look for spiking neuron modules and attention/pooling layers
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if any(keyword in class_name for keyword in ['neuron', 'spike', 'snn', 'lif', 'if']):
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)
    
    return spike_dict, handles


def compute_flops(model, args, input_sample=None):
    """Compute FLOPs for the model using fvcore's official FlopCountAnalysis with custom handlers.
    
    Args:
        model: PyTorch model
        args: Arguments containing model configuration (time_steps, in_channels, batch_size)
        input_sample: Optional real input sample from test dataset. If None, creates dummy input.
    
    Returns:
        flops: Total FLOPs (float)
        params: Total parameters (int)
    """
    import sys
    import io
    
    # Use provided input sample or create dummy input
    if input_sample is None:
        # Create dummy input with proper shape (T, B, C)
        dummy_input = torch.randn(
            args.time_steps,
            args.batch_size,
            args.in_channels,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
    else:
        dummy_input = input_sample
    
    print(f"[FLOPs] Input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}, device: {dummy_input.device}")
    
    # Define handlers for different operation types
    def elementwise_flop_handler(inputs, outputs):
        """Count FLOPs for pure element-wise operations (mul, add, sub)."""
        # 1 FLOP per output element for element-wise operations
        output_value = outputs[0]
        try:
            output_shape = list(output_value.type().sizes())
            if output_shape:
                num_elements = 1
                for dim in output_shape:
                    num_elements *= dim
                print(f"[FLOPs] Elementwise op: shape={output_shape}, flops={num_elements}")
                return num_elements
            else:
                return 0
        except Exception as e:
            print(f"[FLOPs] Error in elementwise handler: {e}")
            return 0
    
    def matmul_flop_handler(inputs, outputs):
        """Count FLOPs for matrix multiplication operations.
        
        For matrix multiply: A @ B where A is (..., m, k) and B is (..., k, n)
        FLOPs = 2 * m * n * k (multiply and accumulate)
        """
        output_value = outputs[0]
        try:
            output_shape = list(output_value.type().sizes())
            if len(output_shape) < 2:
                return 0
                
            # Output shape: (..., m, n)
            m = output_shape[-2]
            n = output_shape[-1]
            
            # Extract k robustly from both inputs
            k = None
            first_input_shape = None
            second_input_shape = None
            
            # Try to get first input shape (A: (..., m, k))
            if len(inputs) > 0:
                try:
                    first_input_shape = list(inputs[0].type().sizes())
                    if len(first_input_shape) >= 2:
                        k = first_input_shape[-1]  # Last dim of A is k
                except Exception as e:
                    print(f"[FLOPs] Could not get first input shape: {e}")
            
            # Fallback: try second input shape (B: (..., k, n))
            if k is None and len(inputs) > 1:
                try:
                    second_input_shape = list(inputs[1].type().sizes())
                    if len(second_input_shape) >= 2:
                        k = second_input_shape[-2]  # Second-to-last dim of B is k
                except Exception as e:
                    print(f"[FLOPs] Could not get second input shape: {e}")
            
            # If still no k, default to 1 (warn this is happening)
            if k is None:
                print(f"[FLOPs] WARNING: Could not extract k from inputs! output_shape={output_shape}, inputs={[first_input_shape, second_input_shape]}")
                k = 1
            
            # Calculate batch size from all dimensions except last 2
            batch_size = 1
            for dim in output_shape[:-2]:
                batch_size *= dim
            
            # FLOPs for matrix multiply: 2 FLOPs per multiply-accumulate (multiply + add)
            # Total: 2 * batch * m * n * k
            flops = 2 * batch_size * m * n * k
            print(f"[FLOPs] MatMul: m={m}, n={n}, k={k}, batch={batch_size}, flops={flops:,.0f} | inputs=[{first_input_shape}, {second_input_shape}]")
            return int(flops)
        except Exception as e:
            print(f"[FLOPs] Error in matmul handler: {e}")
            import traceback
            print(traceback.format_exc())
            return 0
    
    def mean_flop_handler(inputs, outputs):
        """Count FLOPs for mean operation."""
        # Mean: sum all elements (N-1 additions) + divide by N
        # Approximate as 1 FLOP per input element
        output_value = outputs[0]
        try:
            output_shape = list(output_value.type().sizes())
            if output_shape:
                num_elements = 1
                for dim in output_shape:
                    num_elements *= dim
                # Mean is less expensive than matrix ops, so use 1 FLOP per element
                return num_elements
            else:
                return 0
        except Exception as e:
            print(f"[FLOPs] Error in mean handler: {e}")
            return 0
    
    # Suppress stderr to avoid printing unsupported operator warnings
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        # Create FlopCountAnalysis instance
        print("[FLOPs] Creating FlopCountAnalysis instance...")
        flop_counter = FlopCountAnalysis(model, dummy_input)
        print("[FLOPs] FlopCountAnalysis created successfully")
        
        # Register custom handlers for unsupported operators
        print("[FLOPs] Registering custom op handlers...")
        # Element-wise operations: 1 FLOP per element
        flop_counter.set_op_handle("aten::add", elementwise_flop_handler)
        flop_counter.set_op_handle("aten::sub", elementwise_flop_handler)
        flop_counter.set_op_handle("aten::mul", elementwise_flop_handler)
        
        # Matrix operations: proper FLOP counting (2*m*n*k for m,n,k dimensions)
        flop_counter.set_op_handle("aten::mm", matmul_flop_handler)
        flop_counter.set_op_handle("aten::bmm", matmul_flop_handler)
        flop_counter.set_op_handle("aten::matmul", matmul_flop_handler)
        flop_counter.set_op_handle("aten::addmm", matmul_flop_handler)
        flop_counter.set_op_handle("aten::t", lambda inputs, outputs: 0)  # Transpose is free
        
        # Reduction operations
        flop_counter.set_op_handle("aten::mean", mean_flop_handler)
        flop_counter.set_op_handle("aten::sum", mean_flop_handler)
        
        # Structural operations (no actual computation)
        flop_counter.set_op_handle("prim::CallFunction", lambda inputs, outputs: 0)
        print("[FLOPs] Custom handlers registered")
        
        # Compute total FLOPs with custom handlers
        print("[FLOPs] Computing total FLOPs...")
        total_flops = flop_counter.total()
        print(f"[FLOPs] Total FLOPs computed: {total_flops}")
        
    except Exception as e:
        print(f"[FLOPs] ERROR: {e}")
        import traceback
        print(f"[FLOPs] Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        sys.stderr = old_stderr
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[FLOPs] Total parameters: {total_params}")
    
    return total_flops, total_params


def _register_custom_flop_handlers():
    """Deprecated: Custom handlers are now set per-instance in compute_flops()."""
    pass


def calculate_synops(flops, avg_firing_rate):
    """Calculate Synaptic Operations (SynOps) from FLOPs and average firing rate.
    
    SynOps represents the actual number of operations considering spike sparsity.
    SynOps = FLOPs * average_firing_rate
    
    Args:
        flops: Total floating point operations (float)
        avg_firing_rate: Average firing rate across the model (0-1 range typically, but can be >1)
    
    Returns:
        synops: Synaptic operations count (float)
    """
    return flops * avg_firing_rate


def main():

    setup_default_logging()
    args, args_text = _parse_args()

    args.device = "cuda:0"
    _logger.info("Training with a single process on 1 GPUs.")

    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.initial_seed()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random_seed(args.seed)

    model = create_model(
        args.model,
        T=args.time_steps,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        embed_dims=args.dim,
        mlp_ratios=args.mlp_ratio,
        in_channels=args.in_channels,
        depths=args.layer,
        attention_type=args.attention_type,
    )

    _logger.info(f"Creating model {args.model}")
    _logger.info("DOWNSIZE FACTOR: {}".format(args.downsize_factor))
    # _logger.info(str(torchinfo.summary(model, (args.time_steps, 2, args.in_channels))))

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = (
            model.num_classes
        )  # FIXME handle model default vs config num_classes more elegantly

    data_config = resolve_data_config(vars(args), model=model)
    exp_name = "-".join(
        [
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            args.dataset.split("/")[-1],
            f"D{args.dim}",
            f"H{args.num_heads}",
            f"T{args.time_steps}",
            f"L{args.layer}",
            f"MLP{args.mlp_ratio}",
            f"DS{args.downsize_factor}",
        ]
    )
    output_dir = get_outdir("./output/train", exp_name)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"{args.model}.log"), "w"
    )
    file_handler.setFormatter(logging.Formatter(format_str))
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)

    _logger.info(
        f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
    )

    # move model to GPU, enable channels last layout if set
    model.cuda()

    # ============================================================================
    # INFERENCE-ONLY MODE: If --resume is provided with existing path
    # ============================================================================
    if args.resume and os.path.exists(args.resume):
        _logger.info("=" * 100)
        _logger.info("INFERENCE-ONLY MODE ACTIVATED")
        _logger.info(f"Loading pretrained checkpoint: {args.resume}")
        _logger.info("=" * 100)
        
        # Load only the test dataset
        dataset_eval = build_dataset(args, "TEST")

        # Create test dataloader
        loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        
        _logger.info(f"Loaded test dataset: {args.dataset}")
        
        # Load checkpoint
        model.eval()
        resume_checkpoint(model, args.resume, log_info=True)
        
        # Compute and log FLOPs and parameters for inference
        try:
            # Get first sample from test dataset for FLOPs calculation
            first_batch = next(iter(loader_eval))
            input_sample, _ = first_batch
            
            _logger.info(f"Input sample shape before permute: {input_sample.shape}")
            
            # (B, T, C) -> (T, B, C)
            input_sample = input_sample.permute(1, 0, 2).contiguous()
            
            _logger.info(f"Input sample shape after permute: {input_sample.shape}")
            
            input_sample = input_sample.float().cuda()
            
            _logger.info("Starting FLOPs computation...")
            flops, params = compute_flops(model, args, input_sample)
            _logger.info(f"Model FLOPs: {flops:,.0f} | Parameters: {params:,.0f}")
            
        except Exception as e:
            _logger.warning(f"Could not compute FLOPs: {e}")
            import traceback
            _logger.warning(f"Traceback:\n{traceback.format_exc()}")
            flops = None
            params = None
            
        # Create loss function
        validate_loss_fn = nn.CrossEntropyLoss().to(args.device)
        
        # Run 1 epoch of inference on test set to get firing rates
        _logger.info("=" * 100)
        if args.FLOPS:
            _logger.info("RUNNING INFERENCE ON FIRST BATCH ONLY FOR FLOPS/SYNOPS CALCULATION")
            # Create a loader with only the first batch
            first_batch_loader = torch.utils.data.DataLoader(
                dataset_eval,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            first_batch_iter = iter(first_batch_loader)
            first_batch = next(first_batch_iter)
            test_loader_to_use = [first_batch]
        else:
            _logger.info("RUNNING INFERENCE ON ENTIRE TEST DATASET FOR FIRING RATE CALCULATION")
            test_loader_to_use = loader_eval
        _logger.info("=" * 100)
        
        test_metrics = validate(model, test_loader_to_use, validate_loss_fn, args, skip_metrics=args.FLOPS)
        
        # Calculate weighted average firing rate based on tensor sizes
        firing_rate_keys = [k for k in test_metrics.keys() if k.startswith('FiringRate_')]
        tensor_size_keys = [k for k in test_metrics.keys() if k.startswith('TensorSize_')]
        
        # Extract layer names from firing rate keys (e.g., 'FiringRate_layer_name' -> 'layer_name')
        firing_rates_weighted = {}
        total_elements = 0
        weighted_sum = 0.0
        
        for fr_key in firing_rate_keys:
            layer_name = fr_key.replace('FiringRate_', '')
            size_key = f'TensorSize_{layer_name}'
            
            if size_key in test_metrics:
                fr_value = test_metrics[fr_key]
                tensor_size = test_metrics[size_key]
                
                # Accumulate weighted firing rate
                weighted_sum += fr_value * tensor_size
                total_elements += tensor_size
                firing_rates_weighted[layer_name] = (fr_value, tensor_size)
        
        # Calculate weighted average
        avg_firing_rate = weighted_sum / total_elements if total_elements > 0 else 0.0
        
        _logger.info(f"Weighted Average Firing Rate (by tensor size):")
        for layer_name, (fr_val, size) in firing_rates_weighted.items():
            weight_pct = (size / total_elements) * 100
            _logger.info(f"  {layer_name}: FR={fr_val:.6f}, Size={size}, Weight={weight_pct:.2f}%")
        _logger.info(f"Overall Weighted Average Firing Rate: {avg_firing_rate:.6f}")
        
        # Calculate SynOps
        synops = None
        if flops is not None and avg_firing_rate > 0:
            synops = calculate_synops(flops, avg_firing_rate)
            _logger.info(f"Average Firing Rate (across all layers): {avg_firing_rate:.6f}")
            _logger.info(f"Synaptic Operations (SynOps): {synops:,.0f}")
        
        # Export to text file
        export_path = os.path.join(output_dir or ".", "model_info.txt")
        with open(export_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL INFORMATION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model Name: {args.model}\n")
            f.write(f"Dataset: {args.dataset}\n")
            if flops is not None:
                f.write(f"FLOPs: {flops:,.0f}\n")
                f.write(f"Parameters: {params:,.0f}\n")
            if synops is not None:
                f.write(f"Average Firing Rate: {avg_firing_rate:.6f}\n")
                f.write(f"Synaptic Operations (SynOps): {synops:,.0f}\n")
                f.write(f"Compression Ratio (SynOps/FLOPs): {synops/flops:.4f}\n")
            f.write("=" * 80 + "\n")
        _logger.info(f"Model info exported to: {export_path}")
        
        _logger.info("=" * 100)
        _logger.info("INFERENCE RESULTS")
        _logger.info("=" * 100)
        for k, v in test_metrics.items():
            if not k.startswith('FiringRate'):
                _logger.info(f"  {k:15s}: {v:.5f}")
        _logger.info("=" * 100)
        
        return  # Exit after inference
    
    # ============================================================================
    # NORMAL TRAINING MODE
    # ============================================================================
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0

    _logger.info("Scheduled epochs: {}".format(num_epochs))

    transforms_train, transforms_eval = None, None

    dataset_train = build_dataset(args, "TRAIN")
    dataset_val = build_dataset(args, "VAL")
    dataset_eval = build_dataset(args, "TEST")

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    _logger.info("Create dataloader: {}".format(args.dataset))

    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    best_metric = None
    best_epoch = None
    best_metrics_dict = None
    saver = None
    decreasing = False  # F1 score - higher is better
    saver = CheckpointSaver(
        model=model,
        optimizer=optimizer,
        args=args,
        checkpoint_dir=output_dir,
        recovery_dir=output_dir,
        decreasing=decreasing,
    )
    with open(os.path.join(output_dir, "args.yaml"), "w") as f:
        f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
            )
            val_metrics = validate(model, loader_val, validate_loss_fn, args)

            _logger.info("-" * 100)
            _logger.info(f"Epoch {epoch + 1} | VALIDATION SET METRICS")
            _logger.info("-" * 100)
            for k, v in val_metrics.items():
                _logger.info(f"  {k:15s}: {v:.5f}")

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args)

            _logger.info("-" * 100)
            _logger.info(f"Epoch {epoch + 1} | TEST SET METRICS")
            _logger.info("-" * 100)
            for k, v in eval_metrics.items():
                _logger.info(f"  {k:15s}: {v:.5f}")

            if lr_scheduler is not None:
                # step LR based on primary metric for dataset
                primary_metric, metric_name = get_primary_metric(args, val_metrics)
                lr_scheduler.step(epoch + 1, primary_metric)

            if output_dir is not None:
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    os.path.join(output_dir, "summary.csv"),
                    write_header=best_metric is None,
                )

            if saver is not None:
                # Use hierarchical metric comparison
                primary_metric, primary_name = get_primary_metric(args, eval_metrics)
                secondary_metric, secondary_name = get_secondary_metric(args, eval_metrics)
                
                is_better = is_better_metric(eval_metrics, best_metrics_dict, args)
                
                if best_metrics_dict is not None:
                    best_primary, _ = get_primary_metric(args, best_metrics_dict)
                    _logger.info(
                        f"Metric comparison: Current {primary_name}={primary_metric:.5f} "
                        f"vs Best={best_primary:.5f} | Better={is_better}"
                    )

                if is_better:
                    best_metrics_dict = {k: v for k, v in eval_metrics.items()}
                    best_epoch = epoch
                    best_metric = primary_metric
                    
                    # Save the checkpoint
                    saver.save_checkpoint(epoch, metric=primary_metric)
                    
                    _logger.info("-" * 100)
                    _logger.info("*** BEST CHECKPOINT SAVED ***")
                    _logger.info(f"  {primary_name}: {primary_metric:.5f} | {secondary_name}: {secondary_metric:.5f} | Epoch {epoch}")
                    _logger.info("-" * 100)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info("=" * 100)
        _logger.info("TRAINING COMPLETED")
        _logger.info("=" * 100)
        f1 = best_metrics_dict.get("F1", 0)
        top1 = best_metrics_dict.get("top1", 0)
        _logger.info(
            f"  Best F1: {f1:.5f}, Best Top1 Accuracy: {top1:.5f} (Epoch {best_epoch})"
        )
        _logger.info("=" * 100)

    # Load checkpoint and evaluate on test set
    # Priority: --resume arg > best checkpoint from training
    checkpoint_to_load = None
    checkpoint_source = ""
    
    if args.resume and os.path.exists(args.resume):
        checkpoint_to_load = args.resume
        checkpoint_source = "Provided via --resume argument"
    elif best_metric is not None:
        best_checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
        if os.path.exists(best_checkpoint_path):
            checkpoint_to_load = best_checkpoint_path
            checkpoint_source = f"Best checkpoint from training (Epoch {best_epoch})"
    
    if checkpoint_to_load is not None:
        _logger.info(f"Loading checkpoint from {checkpoint_to_load}")
        _logger.info(f"Source: {checkpoint_source}")
        resume_checkpoint(model, checkpoint_to_load, log_info=True)

        # Evaluate on test set
        test_metrics = validate(model, loader_eval, validate_loss_fn, args)

        _logger.info("=" * 100)
        if args.resume and not best_metric:
            _logger.info("TEST RESULTS (Inference Mode - Pretrained Checkpoint)")
        else:
            _logger.info(
                "FINAL TEST RESULTS (Best Checkpoint - Epoch {})".format(best_epoch)
            )
        _logger.info("=" * 100)
        for k, v in test_metrics.items():
            if not k.startswith('FiringRate'):
                _logger.info(f"  {k:15s}: {v:.5f}")
        _logger.info("=" * 100)


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    lr_scheduler=None,
    saver=None,
    amp_autocast=suppress,
):
    sample_number = 0
    start_time = time.time()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # Initialize firing rate tracking
    firing_rate_dict = {}
    non_zero_rate_dict = {}

    model.train()
    functional.reset_net(model)
    
    # Register hooks to capture spike outputs
    spike_dict, hook_handles = register_spike_hooks(model)

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        # (B, T, C) -> (T, B, C)
        input = input.permute(1, 0, 2).contiguous()
        input = input.float().cuda(non_blocking=True)
        target = target.long().cuda(non_blocking=True)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)
        
        # Calculate firing rate statistics
        if spike_dict:
            batch_fr, batch_nz, batch_sizes = calc_firing_rate_stats(spike_dict)
            for layer_name, fr_val in batch_fr.items():
                if layer_name not in firing_rate_dict:
                    firing_rate_dict[layer_name] = []
                firing_rate_dict[layer_name].append(fr_val)
            for layer_name, nz_val in batch_nz.items():
                if layer_name not in non_zero_rate_dict:
                    non_zero_rate_dict[layer_name] = []
                non_zero_rate_dict[layer_name].append(nz_val)
        
        sample_number += input.shape[0]
        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        functional.reset_net(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or batch_idx % 10 == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _logger.info(
                "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                "LR: {lr:.3e}  "
                "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                    epoch,
                    batch_idx,
                    len(loader),
                    100.0 * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m,
                )
            )
            
            # Log firing rates periodically
            if firing_rate_dict:
                _logger.info("  Firing Rates per Layer:")
                for layer_name, fr_vals in firing_rate_dict.items():
                    avg_fr = np.mean(fr_vals) if fr_vals else 0
                    _logger.info(f"    {layer_name}: FR={avg_fr:.6f}")

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()
    
    # Remove spike hooks
    for handle in hook_handles:
        handle.remove()

    _logger.info(f"samples / s = {sample_number / (time.time() - start_time): .3f}")
    
    # Log epoch firing rates
    if firing_rate_dict:
        _logger.info(f"Epoch {epoch} - Average Firing Rates:")
        for layer_name, fr_vals in firing_rate_dict.items():
            avg_fr = np.mean(fr_vals) if fr_vals else 0
            nz_vals = non_zero_rate_dict.get(layer_name, [])
            avg_nz = np.mean(nz_vals) if nz_vals else 0
            _logger.info(f"  {layer_name}: Firing Rate={avg_fr:.6f}, Non-zero Rate={avg_nz:.6f}")
    
    return OrderedDict([("loss", losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix="", skip_metrics=False):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    functional.reset_net(model)

    # Collect predictions and targets for comprehensive metrics
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Initialize firing rate tracking
    firing_rate_dict = {}
    non_zero_rate_dict = {}
    tensor_size_dict = {}
    
    # Register hooks to capture spike outputs
    spike_dict, hook_handles = register_spike_hooks(model)

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # (B, T, C) -> (T, B, C)
            input = input.permute(1, 0, 2).contiguous()
            input = input.float().cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            last_batch = batch_idx == last_idx

            output = model(input)
            
            # Calculate firing rate statistics
            if spike_dict:
                batch_fr, batch_nz, batch_sizes = calc_firing_rate_stats(spike_dict)
                for layer_name, fr_val in batch_fr.items():
                    if layer_name not in firing_rate_dict:
                        firing_rate_dict[layer_name] = []
                    firing_rate_dict[layer_name].append(fr_val)
                for layer_name, nz_val in batch_nz.items():
                    if layer_name not in non_zero_rate_dict:
                        non_zero_rate_dict[layer_name] = []
                    non_zero_rate_dict[layer_name].append(nz_val)
                # Store tensor size (same for all batches from same layer)
                for layer_name, size in batch_sizes.items():
                    if layer_name not in tensor_size_dict:
                        tensor_size_dict[layer_name] = size

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            # Collect predictions and targets for metrics calculation
            if not skip_metrics:
                preds = torch.nn.functional.softmax(output, dim=1)
                all_preds.append(output.detach().cpu())
                all_probs.append(preds.detach().cpu())
                all_targets.append(target.detach().cpu())

            batch_time_m.update(time.time() - end)
            end = time.time()

            if last_batch or batch_idx % 10 == 0:
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )
                
                # Log firing rates periodically
                if firing_rate_dict:
                    _logger.info("Firing Rates per Layer:")
                    for layer_name, fr_vals in firing_rate_dict.items():
                        avg_fr = np.mean(fr_vals) if fr_vals else 0
                        _logger.info(f"    {layer_name}: FR={avg_fr:.6f}")
    
    # Remove spike hooks
    for handle in hook_handles:
        handle.remove()

    # Skip metrics calculation if requested
    if skip_metrics:
        metrics = OrderedDict([("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)])
        
        # Add firing rate statistics to metrics
        if firing_rate_dict:
            _logger.info("Validation - Average Firing Rates per Layer:")
            for layer_name, fr_vals in firing_rate_dict.items():
                avg_fr = np.mean(fr_vals) if fr_vals else 0
                nz_vals = non_zero_rate_dict.get(layer_name, [])
                avg_nz = np.mean(nz_vals) if nz_vals else 0
                metrics[f"FiringRate_{layer_name}"] = avg_fr
                metrics[f"TensorSize_{layer_name}"] = tensor_size_dict.get(layer_name, 0)
                _logger.info(f"  {layer_name}: FR={avg_fr:.6f}, TensorSize={tensor_size_dict.get(layer_name, 0)}")
        
        return metrics

    # Calculate comprehensive metrics
    preds_cat = torch.cat(all_preds, 0)
    probs_cat = torch.cat(all_probs, 0)
    targets_cat = torch.cat(all_targets, 0)

    # Convert to numpy
    predictions = torch.argmax(probs_cat, dim=1).numpy()
    probs_numpy = probs_cat.numpy()
    targets_numpy = targets_cat.flatten().numpy()
    
    # Debug: Log class distribution
    unique_preds, counts_preds = np.unique(predictions, return_counts=True)
    unique_targets, counts_targets = np.unique(targets_numpy, return_counts=True)
    _logger.info(f"Prediction distribution: {dict(zip(unique_preds, counts_preds))}")
    _logger.info(f"Target distribution: {dict(zip(unique_targets, counts_targets))}")

    # One-hot encode targets for metrics that require it
    targets_onehot = (
        torch.nn.functional.one_hot(
            targets_cat.reshape(
                -1,
            ).to(torch.long),
            num_classes=args.num_classes,
        )
        .float()
        .numpy()
    )

    # Calculate metrics
    metrics_dict = {
            "Accuracy": accuracy_score(targets_numpy, predictions),
            "Precision": precision_score(
                targets_numpy, predictions, average="macro", zero_division=0
            ),
            "Recall": recall_score(
                targets_numpy, predictions, average="macro", zero_division=0
            ),
            "F1": f1_score(targets_numpy, predictions, average="macro", zero_division=0),
            "AUROC": roc_auc_score(
                targets_onehot, probs_numpy, multi_class="ovr", average="macro"
            ),
            "AUPRC": average_precision_score(targets_onehot, probs_numpy, average="macro"),
        }
    
    # Add firing rate statistics to metrics
    if firing_rate_dict:
        _logger.info("Validation - Average Firing Rates per Layer:")
        for layer_name, fr_vals in firing_rate_dict.items():
            avg_fr = np.mean(fr_vals) if fr_vals else 0
            nz_vals = non_zero_rate_dict.get(layer_name, [])
            avg_nz = np.mean(nz_vals) if nz_vals else 0
            metrics_dict[f"FiringRate_{layer_name}"] = avg_fr
            metrics_dict[f"TensorSize_{layer_name}"] = tensor_size_dict.get(layer_name, 0)
            # metrics_dict[f"NonZeroRate_{layer_name}"] = avg_nz
            _logger.info(f"  {layer_name}: FR={avg_fr:.6f}, TensorSize={tensor_size_dict.get(layer_name, 0)}")

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
        + list(metrics_dict.items())
    )

    return metrics


if __name__ == "__main__":
    main()
