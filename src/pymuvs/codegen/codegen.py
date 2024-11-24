import sympy as sp
import numpy as np

from ..link import Model


def vector_to_cppfn(m: sp.Matrix, name: str, indent: str = '\t', **kwargs):
    variables = {v for varlist in kwargs.values() for v in varlist}
    assert m.free_symbols.issubset(
        variables), "All free symbols in the matrix must be in the variables list"

    identifier = "m"
    while identifier in variables or identifier == name:
        identifier += str(np.random.randint(0, 9))

    code = f"Eigen::VectorXd {name}("
    code += ", ".join([f"Eigen::VectorXd {vname}" for vname in kwargs.keys()])
    code += ") {\n"
    for vname, vlist in kwargs.items():
        for i, v in enumerate(vlist):
            code += f"{indent}auto {v} = {vname}({i});\n"
    code += f"{indent}Eigen::VectorXd {identifier}({m.rows}, {m.cols});\n"
    rows, cols = m.shape
    for r in range(rows):
        expr = sp.ccode(m[r, 0])
        code += f"{indent}{identifier}({r}) = {expr};\n"
    code += f"{indent}return {identifier};\n" + "}"

    return code


def matrix_to_cppfn(m: sp.Matrix, name: str, indent: str = '\t', **kwargs):
    variables = {v for varlist in kwargs.values() for v in varlist}
    assert m.free_symbols.issubset(
        variables), "All free symbols in the matrix must be in the variables list"

    identifier = "m"
    while identifier in variables or identifier == name:
        identifier += str(np.random.randint(0, 9))

    code = f"Eigen::MatrixXd {name}("
    code += ", ".join([f"Eigen::VectorXd {vname}" for vname in kwargs.keys()])
    code += ") {\n"
    for vname, vlist in kwargs.items():
        for i, v in enumerate(vlist):
            code += f"{indent}auto {v} = {vname}({i});\n"
    code += f"{indent}Eigen::MatrixXd {identifier}({m.rows}, {m.cols});\n"
    rows, cols = m.shape
    for r in range(rows):
        for c in range(cols):
            expr = sp.ccode(m[r, c])
            code += f"{indent}{identifier}({r}, {c}) = {expr};\n"
    code += f"{indent}return {identifier};\n" + "}"

    return code


def to_cppfn(m: sp.Matrix, name: str, indent: str = '\t', **kwargs):
    if m.shape[0] == 1 or m.shape[1] == 1:
        return vector_to_cppfn(m, name, indent, **kwargs)
    return matrix_to_cppfn(m, name, indent, **kwargs)


def model_to_cpp(m: Model, indent: str = '\t') -> str:
    header = "#include <Eigen/Dense>\n#include <math.h>\n"
    mstr = to_cppfn(m.M, "M", q=m.params, indent=indent)
    cstr = to_cppfn(m.C, "C", q=m.params, dq=m.diff_params, indent=indent)
    dstr = to_cppfn(m.D, "D", q=m.params, dq=m.diff_params, indent=indent)
    gstr = to_cppfn(m.g, "g", q=m.params, indent=indent)

    # include function to get ddq
    # TODO: include J, Jf, B, u, z
    ddqfn = """Eigen::VectorXd ddq(Eigen::VectorXd q, Eigen::VectorXd dq) {\n"""
    ddqfn += f"{indent}auto mM = M(q);\n"
    ddqfn += f"{indent}auto mC = C(q, dq);\n"
    ddqfn += f"{indent}auto mD = D(q, dq);\n"
    ddqfn += f"{indent}auto mg = g(q);\n"
    ddqfn += f"{indent}return mM.inverse() * (mg - mC * dq - mD * dq);\n" + "}"

    code = header + "\n\n" + "namespace Model {" + "\n\n" + mstr + "\n\n" + \
        cstr + "\n\n" + dstr + "\n\n" + gstr + "\n\n" + ddqfn + "\n\n" + \
        "} // namespace Model\n"
    print(code)
    return code
