# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pkgutil
import importlib

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.mps
except ImportError:
    pass


class MPS_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'MPS'
        self._communication_backend_name = 'mps'

        # begin initialize for create_op_builder()
        # put all valid class name <--> class type mapping into class_dict
        op_builder_dir = self.op_builder_dir()
        op_builder_module = importlib.import_module(op_builder_dir)
        for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(op_builder_module.__file__)]):
            # avoid self references
            if module_name != 'all_ops' and module_name != 'builder':
                module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                for member_name in module.__dir__():
                    if member_name.endswith(
                            'Builder'
                    ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder" and member_name != "TorchCPUOpBuilder":  # avoid abstract classes
                        if not member_name in self.class_dict:
                            self.class_dict[member_name] = getattr(module, member_name)
        # end initialize for create_op_builder()


    def synchronize(self, device_index=None):
        return torch.mps.synchronize()

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        return torch.cuda.set_rng_state(new_state)

    def get_rng_state(self, device_index=None):
        return torch.mps.set_rng_state()

    def manual_seed(self, seed):
        return torch.mps.manual_seed(seed)


    # Memory management
    def empty_cache(self):
        return torch.mps.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.mps.current_allocated_memory()

    def max_memory_allocated(self, device_index=None):
        return torch.mps.driver_allocated_memory()

    def communication_backend_name(self):
        return self._communication_backend_name

    def pin_memory(self, tensor):
        return tensor.pin_memory()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('mps'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401
            return "op_builder"
        except ImportError:
            return "deepspeed.ops.op_builder"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
    # this dict will be filled at init stage
    class_dict = {}

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        else:
            return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension
