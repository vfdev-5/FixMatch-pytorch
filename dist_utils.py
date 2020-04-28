import numbers

import torch
import torch.distributed as dist

try:
    import torch_xla.core.xla_model as xm
    has_xla_support = True
except ImportError:
    has_xla_support = False


def is_gpu_distributed():
    return dist.is_available() and dist.is_initialized()


def is_tpu_distributed():
    return has_xla_support and xm.xrt_world_size() > 1


def get_world_size():
    if is_gpu_distributed():
        return dist.get_world_size()
    elif is_tpu_distributed():
        return xm.xrt_world_size()
    else:
        return 1


def get_rank():
    if is_gpu_distributed():
        return dist.get_rank()
    elif is_tpu_distributed():
        return xm.get_ordinal()
    else:
        return 0


def get_num_proc_per_node():
    if is_gpu_distributed():
        return torch.cuda.device_count()
    elif is_tpu_distributed():
        return xm.get_xla_supported_devices()
    else:
        return 1


def device(default_value):
    if is_gpu_distributed():
        return torch.cuda.current_device()
    elif is_tpu_distributed():
        return xm.xla_device()
    return default_value


def initialize():
    if has_xla_support:
        xm.rendezvous('init')
    else:
        torch.backends.cudnn.benchmark = True
        dist.init_process_group("nccl", init_method="env://")


def finalize():
    if not has_xla_support:
        dist.destroy_process_group()


def _tpu_sync_all_reduce(self, tensor):
    tensor_to_number = False
    if isinstance(tensor, numbers.Number):
        tensor = torch.tensor(tensor, device=self._device, dtype=torch.float)
        tensor_to_number = True

    if isinstance(tensor, torch.Tensor):
        # check if the tensor is at specified device
        if tensor.device != self._device:
            tensor = tensor.to(self._device)
    else:
        raise TypeError("Unhandled input type {}".format(type(tensor)))

    # synchronize and reduce
    xm.all_reduce("sum", [tensor, ])

    if tensor_to_number:
        return tensor.item()
    return tensor


def _temporary_ignite_metrics_patch():
    # until merged https://github.com/pytorch/ignite/issues/992
    if is_tpu_distributed():
        from ignite.metrics import Metric
        Metric._sync_all_reduce = _tpu_sync_all_reduce


_temporary_ignite_metrics_patch()
