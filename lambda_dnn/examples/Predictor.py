import numpy as np
import math
import time
import random

def f_l(m,n,B_f_s,B_f_p,R):
    k= ((n_t //n) //b_l_f) * e
    T_pre = (d_t / (n * B_f_s) + k * (b_l_f / R + (2 * d_m) / B_f_p)) 
    cost = (((m/1024)*n*0.0000166667+2.6*10**(-5))*T_pre)
    return  T_pre,cost
#fix global batch size, and  b_l_f * n>b_g_max
def f_g(m,n,B_f_s,B_f_p,R):
    k= (n_t //b_g_max) * e
    T_pre = (d_t / (n * B_f_s) + k * (b_g_max / (n*R)+ (2 * d_m) / B_f_p)) #given global batch size
    cost = (((m/1024)*n*0.0000166667+2.6*10**(-5))*T_pre)
    return  T_pre,cost

def lambdadnn(T_o):
    min_cost = 1000
    T_pre = 10000
    r_m = 128
    r_n = 1
    B_f_p = 80
    if n_divide>=B_pmax//B_f_p: # only train with fixed local batch size is enough
        for m in range(min_memory, 3072, 64):
            if m < 1792:
                B_f_s = 0.002228 * m + 16.12  # bandwidth of s3
            else:
                B_f_s = 0.002228 * 1792 + 16.12
            nlow = math.ceil((1 / T_o) * (d_t / B_f_s + (n_t * e) / R + (2 * d_m * n_t * e) / (b_l_f * B_f_p)))
            n_upp = math.floor(B_pmax//B_f_p)
            for n in range(nlow, n_upp + 1):
                if n > 16:
                    B_f_p = B_pmax / n
                T_predict_fl = f_l(m, n, B_f_s, B_f_p)[0]
                cost_predict_fl = f_l(m, n, B_f_s, B_f_p)[1]
                if T_predict_fl < T_o and cost_predict_fl < min_cost:
                    r_m = m
                    r_n  = n
                    T_pre = T_predict_fl
                    min_cost = cost_predict_fl
                    break
                else:
                    continue
    if n_divide < B_pmax // B_f_p:
        for m in range(min_memory, 3072, 64):
            if m < 1792:
                B_f_s = 0.002228 * m + 16.12  # bandwidth of s3
            else:
                B_f_s = 0.002228 * 1792 + 16.12
            nlow_bl = math.ceil((1 / T_o) * (d_t / B_f_s + (n_t * e) / R + (2 * d_m * n_t * e) / (b_l_f * B_f_p)))
            #print("nlow_bl=",nlow_bl)
            if nlow_bl >= n_divide:  # can not training if fix local batchs size
                nlow_To_bg = math.ceil((b_g_max * B_f_p * (d_t * R + n_t * e * B_f_s)) / (
                        B_f_s * R * (T_o * b_g_max * B_f_p - n_t * e * 2 * d_m)))  # given b_global
                nlow_bg = max(nlow_To_bg, nlow_bl)
                n_upp = math.floor(B_pmax//B_f_p)
                for n in range(nlow_bg, n_upp + 1):
                    B_f_p = 80
                    if n > 16:
                        B_f_p = B_pmax / n
                    T_predict_fg = f_g(m, n, B_f_s, B_f_p)[0]
                    cost_predict_fg = f_g(m, n, B_f_s, B_f_p)[1]
                    if T_predict_fg < T_o and cost_predict_fg < min_cost:
                        r_m = m
                        r_n = n
                        T_pre = T_predict_fg
                        min_cost = cost_predict_fg
                        break
                    else:
                        continue
            if nlow_bl < n_divide:
                nupp = math.floor(B_pmax // B_f_p)
                for n in range(nlow_bl, nupp+1):
                    B_f_p = 80
                    if n > 16:
                        B_f_p = B_pmax / n
                    if n <= n_divide:
                        T_predict_fl = f_l(m, n, B_f_s, B_f_p)[0]
                        cost_predict_fl = f_l(m, n, B_f_s, B_f_p)[1]
                        if T_predict_fl < T_o and cost_predict_fl < min_cost:
                            r_m = m
                            r_n = n
                            T_pre = T_predict_fl
                            min_cost = cost_predict_fl
                            break
                        else:
                            continue
                    if n > n_divide:
                        nlow_T_o = math.ceil((b_g_max * B_f_p * (d_t * R + n_t * e * B_f_s)) / (
                                B_f_s * R * (T_o * b_g_max * B_f_p - n_t * e * 2 * d_m)))  # given b_global
                        nlow = max(n_divide, nlow_T_o)
                        n = nlow
                        T_predict_fg = f_g(m, n, B_f_s, B_f_p)[0]
                        cost_predict_fg = f_g(m, n, B_f_s, B_f_p)[1]
                        if T_predict_fg < T_o and cost_predict_fg < min_cost:
                            r_m = m
                            r_n = n
                            T_pre = T_predict_fg
                            min_cost = cost_predict_fg
                            break
                        else:
                            continue
    return T_pre,min_cost,r_m,r_n
    
