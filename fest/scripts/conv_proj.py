import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from fest import get_spacetime_errornorm

# 你原先定义的参数和函数
mu = 0.001

def bc_expr(x):
    return cos(2*pi*x)

def h(u, v, mu=mu):
    return (u.dx(1) * v + u.dx(0) * mu * v.dx(0)) * dx

def L(u, v):
    return Constant(0.0) * v * dx

def exact(x, t):
    return cos(2*pi*x)*exp(-4*(pi**2) * mu * t)

def refine_spacetime_alternately(sp_init=10, t_init=1, t_end=0.5, deg=3, 
                                 bc_expr=None, h=None, L=None, exact=None,
                                 num_iter_mode='step',  # 'step'表示solve-by-lapse, 'once'表示solve-at-once
                                 threshold1=1.0,        # 阈值1：用于判定“空间加密的收敛收益是否足够”
                                 threshold2=1e-12,      # 阈值2：最终目标误差
                                 max_iter=40):
    """
    一直 refine 空间剖分, 直到 consecutive error 变化量 < threshold1 后，
    再去 refine 时间剖分，直到 error < threshold2.
    """
    sp_res = sp_init
    t_res = t_init
    
    sp_history = []
    t_history = []
    err_history = []

    # 先算一个初始误差
    err_old = get_spacetime_errornorm(
        sp_res=sp_res,
        t_res=t_res,
        t_end=t_end,
        deg=deg,
        bc_expr=bc_expr,
        h=h,
        L=L,
        exact=exact,
        num_iter=(t_res if num_iter_mode == 'step' else 1)
    )
    err_history.append(err_old)

    state = 'space'   # 当前在哪个阶段：先做空间加密
    iteration = 0

    while iteration < max_iter:
        iteration += 1

        if state == 'space':
            # 加密空间剖分
            sp_res += 10  # 你可以改成其他加密策略，例如乘2等等

            # 计算新误差
            err_new = get_spacetime_errornorm(
                sp_res=sp_res,
                t_res=t_res,
                t_end=t_end,
                deg=deg,
                bc_expr=bc_expr,
                h=h,
                L=L,
                exact=exact,
                num_iter=(t_res if num_iter_mode == 'step' else 1)
            )

            sp_history.append(sp_res)
            t_history.append(t_res)
            err_history.append(err_new)

            print(f"[{iteration}] SPACE refine => sp_res={sp_res}, t_res={t_res}, err={err_new}")

            # 判断这次误差变化量是否足够小
            if abs(err_new - err_old) < threshold1:
                # 如果已经不足以继续大幅降低误差，则切换到 refine time
                state = 'time'

            err_old = err_new  # 更新历史误差以便下一次比较

        else:  # state == 'time'
            # 加密时间剖分
            t_res += 1  # 你可改成 t_res *= 2，或其它策略

            # 计算新误差
            err_new = get_spacetime_errornorm(
                sp_res=sp_res,
                t_res=t_res,
                t_end=t_end,
                deg=deg,
                bc_expr=bc_expr,
                h=h,
                L=L,
                exact=exact,
                num_iter=(t_res if num_iter_mode == 'step' else 1)
            )

            sp_history.append(sp_res)
            t_history.append(t_res)
            err_history.append(err_new)

            print(f"[{iteration}] TIME refine => sp_res={sp_res}, t_res={t_res}, err={err_new}")

            # 如果误差已经小于 threshold2，则停止
            if err_new < threshold2:
                print(f"Final error < {threshold2}, stopping iteration.")
                break

            err_old = err_new  # 更新历史误差

    print("Refinement process finished.")
    print(f"Final sp_res={sp_res}, t_res={t_res}, final_err={err_old}, iteration={iteration}")
    return sp_history, t_history, err_history


# ------------------ 以下为示例调用并画图 ------------------
if __name__ == "__main__":
    sp_hist, t_hist, err_hist = refine_spacetime_alternately(
        sp_init=10,
        t_init=1,
        t_end=0.5,
        deg=3,
        bc_expr=bc_expr,
        h=h,
        L=L,
        exact=exact,
        num_iter_mode='step',
        threshold1=1.0,   # 当连续迭代的误差变化小于1时，转去 refine 时间
        threshold2=1e-12, # 最终目标误差
        max_iter=40
    )

    # 简单绘图
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(range(len(err_hist)), err_hist, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error (log scale)")
    ax.set_title("Space -> (once difference < threshold1) -> Time refinement")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()