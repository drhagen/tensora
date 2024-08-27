__all__ = ["compile_module"]

import llvmlite.binding as llvm

from ..codegen import ir_to_llvm
from ..ir.ast import Module
from ._initialize_llvm import target


def compile_module(module: Module) -> llvm.ExecutionEngine:
    llvm_ir = ir_to_llvm(module)

    # Compile the module
    llvm_module = llvm.parse_assembly(str(llvm_ir))
    llvm_module.verify()

    # Create target machine
    # We have to recreate this for every module because create_mcjit_compiler
    # takes ownership of target_machine and frees it when the engine goes out of scope
    target_machine = target.create_target_machine()

    # Create execution engine
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

    # Add the module to the engine and make sure it is ready for execution
    engine.add_module(llvm_module)
    engine.finalize_object()
    engine.run_static_constructors()

    return engine
