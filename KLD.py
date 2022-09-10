import numpy as np


def calc_KL(P, Q):
    # 获取概率分布
    hist_P, bins_P = np.histogram(P)
    hist_Q, bins_Q = np.histogram(Q)

    # 概率分布归一化
    pdf_P = hist_P / (np.sum(hist_P) + 1e-8)
    pdf_Q = hist_Q / (np.sum(hist_Q) + 1e-8)

    # 计算 KL Divergence
    kld_PQ = np.sum(pdf_P * (np.log(pdf_P + 1e-8) - np.log(pdf_Q + 1e-8)))  # P * log(P/Q)

    return pdf_P, pdf_Q, kld_PQ


if __name__ == '__main__':
    batch_size, out_dim = 128, 1
    P = np.random.rand(batch_size, out_dim)
    Q = np.random.rand(batch_size, out_dim)
    pdf_P, pdf_Q, kld_PQ = calc_KL(P, Q)
    print(pdf_P, pdf_Q, kld_PQ)
    print(pdf_P.shape, pdf_Q.shape, kld_PQ.shape)