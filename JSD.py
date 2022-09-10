import numpy as np


def kl_divergence(pdf_P, pdf_Q):
    kld_PQ = np.sum(pdf_P * (np.log(pdf_P + 1e-8) - np.log(pdf_Q + 1e-8)))
    return kld_PQ


def calc_JS(P, Q):
    # 获取概率分布
    hist_P, bins_P = np.histogram(P)
    hist_Q, bins_Q = np.histogram(Q)

    # 概率分布归一化
    pdf_P = hist_P / (np.sum(hist_P) + 1e-8)
    pdf_Q = hist_Q / (np.sum(hist_Q) + 1e-8)

    # 计算 JS Divergence
    m = 0.5 * (pdf_P + pdf_Q)
    jsd_pq = 0.5 * kl_divergence(pdf_P, m) + 0.5 * kl_divergence(pdf_Q, m)
    return pdf_P, pdf_Q, jsd_pq


if __name__ == '__main__':
    batch_size, out_dim = 128, 1
    P = np.random.rand(batch_size, out_dim)
    Q = np.random.rand(batch_size, out_dim)
    pdf_P, pdf_Q, jsd_PQ = calc_JS(P, Q)
    print(pdf_P, pdf_Q, jsd_PQ)
    print(pdf_P.shape, pdf_Q.shape, jsd_PQ.shape)