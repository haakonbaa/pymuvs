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

    declaration = f"Eigen::VectorXd {name}("
    declaration += ", ".join(
        [f"Eigen::VectorXd {vname}" for vname in kwargs.keys()])
    declaration += ")"
    code = declaration + " {\n"
    for vname, vlist in kwargs.items():
        for i, v in enumerate(vlist):
            code += f"{indent}double {v} = {vname}({i});\n"
    code += f"{indent}Eigen::VectorXd {identifier}({m.rows}, {m.cols});\n"
    rows, cols = m.shape
    for r in range(rows):
        expr = sp.ccode(m[r, 0])
        code += f"{indent}{identifier}({r}) = {expr};\n"
    code += f"{indent}return {identifier};\n" + "}"

    return code, declaration


def matrix_to_cppfn(m: sp.Matrix, name: str, indent: str = '\t', **kwargs):
    variables = {v for varlist in kwargs.values() for v in varlist}
    assert m.free_symbols.issubset(
        variables), "All free symbols in the matrix must be in the variables list"

    identifier = "m"
    while identifier in variables or identifier == name:
        identifier += str(np.random.randint(0, 9))

    declaration = f"Eigen::MatrixXd {name}("
    declaration += ", ".join(
        [f"Eigen::VectorXd {vname}" for vname in kwargs.keys()])
    declaration += ")"
    code = declaration + " {\n"
    for vname, vlist in kwargs.items():
        for i, v in enumerate(vlist):
            code += f"{indent}double {v} = {vname}({i});\n"
    code += f"{indent}Eigen::MatrixXd {identifier}({m.rows}, {m.cols});\n"
    rows, cols = m.shape
    for r in range(rows):
        for c in range(cols):
            expr = sp.ccode(m[r, c])
            code += f"{indent}{identifier}({r}, {c}) = {expr};\n"
    code += f"{indent}return {identifier};\n" + "}"

    return code, declaration


def to_cppfn(m: sp.Matrix, name: str, indent: str = '\t', **kwargs):
    if m.shape[0] == 1 or m.shape[1] == 1:
        return vector_to_cppfn(m, name, indent, **kwargs)
    return matrix_to_cppfn(m, name, indent, **kwargs)


def model_to_cpp(m: Model, indent: str = '\t') -> str:
    header = "#include <Eigen/Dense>\n#include <math.h>\n#include <stdexcept>\n\n"
    header += "/*" + Model.__doc__ + "\n"
    header += f" q = {m.params}\n"
    header += f"dq = {m.diff_params}\n"
    header += f" z = {m.inputs}\n"
    uvec = [m.u[i, 0] for i in range(m.u.shape[0])]
    header += f" u = {uvec}\n"
    header += "*/\n\n"  # end of headerfile comment
    header += "namespace Model {\n\n"
    header += f"constexpr int Nn  = {m.M.shape[0]};\n"
    header += f"constexpr int Nj  = {m.J.shape[0]};\n"
    header += f"constexpr int Nb  = {m.B.shape[0]};\n"
    header += f"constexpr int Nm  = {m.u.shape[0]};\n"
    header += f"constexpr int Nz  = {len(m.inputs)}; // number of inputs\n"
    header += f"constexpr int Nt  = {len(m.transforms)}; // number of transforms\n\n"
    body = ""

    mstr, declrm = to_cppfn(m.M, "M", q=m.params, indent=indent)
    header += declrm + ";\n"
    body += mstr + "\n"

    cstr, declrc = to_cppfn(m.C, "C", q=m.params,
                            dq=m.diff_params, indent=indent)
    header += declrc + ";\n"
    body += cstr + "\n"

    dstr, declrd = to_cppfn(m.D, "D", q=m.params,
                            dq=m.diff_params, indent=indent)
    header += declrd + ";\n"
    body += dstr + "\n"

    gstr, declrg = to_cppfn(m.g, "g", q=m.params, indent=indent)
    header += declrg + ";\n"
    body += gstr + "\n"

    Jstr, declrJ = to_cppfn(m.J, "J", q=m.params, indent=indent)
    header += declrJ + ";\n"
    body += Jstr + "\n"

    Jfstr, declrJf = to_cppfn(m.Jf, "Jf", q=m.params, indent=indent)
    header += declrJf + ";\n"
    body += Jfstr + "\n"

    Bstr, declrB = to_cppfn(m.B, "B", indent=indent)
    header += declrB + ";\n"
    body += Bstr + "\n"

    ustr, declru = to_cppfn(m.u, "u", z=m.inputs, indent=indent)
    header += declru + ";\n"
    body += ustr + "\n"

    Justr, declrJu = to_cppfn(m.Ju, "Ju", z=m.inputs, indent=indent)
    header += declrJu + ";\n"
    body += Justr + "\n"

    for i, transform in enumerate(m.transforms):
        x, y, z = sp.symbols('x y z')
        tfn = transform.apply(sp.Matrix([x, y, z]))
        kwargs = {f"x_{i}":[x, y, z]}
        fistr, declrfi = to_cppfn(tfn, f"T{i}_to_b", q=m.params, **kwargs, indent=indent)
        header += declrfi + ";\n"
        body += fistr + "\n"


    ddqfn = """Eigen::VectorXd ddq(Eigen::VectorXd q, Eigen::VectorXd dq, Eigen::VectorXd z) {\n"""
    ddqfn += f"{indent}Eigen::MatrixXd mM = M(q);\n"
    ddqfn += f"{indent}Eigen::MatrixXd mC = C(q, dq);\n"
    ddqfn += f"{indent}Eigen::MatrixXd mD = D(q, dq);\n"
    ddqfn += f"{indent}Eigen::VectorXd mg = g(q);\n"
    ddqfn += f"{indent}Eigen::MatrixXd mJf = Jf(q);\n"
    ddqfn += f"{indent}Eigen::MatrixXd mB = B();\n"
    ddqfn += f"{indent}Eigen::VectorXd mu = u(z);\n"
    ddqfn += f"{indent}return mM.inverse() * (mJf * (mB * mu) - mC * dq - mD * dq - mg);\n" + "}"
    header += "Eigen::VectorXd ddq(Eigen::VectorXd q, Eigen::VectorXd dq, Eigen::VectorXd z);\n"
    body += ddqfn + "\n"

    # u_to_z function
    if m.u_to_z is None:
        u_to_z = "Eigen::VectorXd u_to_z(Eigen::VectorXd u) {\n"
        u_to_z += f'{indent}throw std::runtime_error("Error: u_to_z not implemented");\n'
        u_to_z += f"{indent}return" + " Eigen::VectorXd();\n" + "}"
        header += "Eigen::VectorXd u_to_z(Eigen::VectorXd u);"
        body += u_to_z + "\n"
    else:
        u_to_z, declrutoz = to_cppfn(
            m.u_to_z, "u_to_z", u=m.uvars, indent=indent)
        header += declrutoz + ";\n"
        body += u_to_z + "\n"

    header += "\n\n"
    body += "\n} // namespace Model\n"

    print(f"{header=}")
    print(f"\n\n\n\n\n")
    print(f"{body=}")

    headerfile = header + "} // namespace Model\n"
    cppfile = r'#include "model.h"' + "\n\n"
    cppfile += "namespace Model { // namespace Model\n" + body + "\n"

    return header + body, headerfile, cppfile
