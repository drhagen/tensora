__all__ = ["target"]

import llvmlite.binding as llvm

# Initialize the LLVM
# https://llvmlite.readthedocs.io/en/latest/user-guide/binding/examples.html
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# Create the target representing the current host
target = llvm.Target.from_default_triple()
